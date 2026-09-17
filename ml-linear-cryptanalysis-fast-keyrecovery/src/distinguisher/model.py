"""The neural distinguisher ND_r^t.

The paper uses Gohr's (2019) residual network with only the input layer resized
to the sample dimension DIM = t.  We keep the residual architecture but make it
small and dense rather than convolutional:

* the paper's own Appendix A / Table 7 shows the network family barely matters
  for this task (ResNet, SENet and DenseNet agree to within 0.2 percentage
  points), and
* the samples here are flat t-bit vectors of i.i.d. Bernoulli variables with no
  spatial structure for convolutions to exploit, so a residual MLP is the
  faithful dense analogue.

Crucially, our contribution is the *search strategy*, not the distinguisher --
the baseline and the guided search use one and the same trained network, so the
architecture choice cancels out of the comparison.  :func:`optimal_accuracy` in
``data_gen`` gives the Bayes ceiling that this network is checked against.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.bn1(self.fc1(x)))
        h = self.bn2(self.fc2(h))
        return self.act(x + h)


class NeuralDistinguisher(nn.Module):
    """ND_r^t: maps a t-bit sample to a probability that it is a positive."""

    def __init__(self, dim, width=32, depth=2):
        super().__init__()
        self.dim = dim
        self.width = width
        self.depth = depth
        self.stem = nn.Sequential(
            nn.Linear(dim, width), nn.BatchNorm1d(width), nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        return self.head(h).squeeze(-1)      # logits

    # -- inference helpers -------------------------------------------------
    @torch.no_grad()
    def logit(self, x):
        """Raw logits for a numpy/torch batch (n, t).  Network in eval mode."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
        return self.forward(x)

    @torch.no_grad()
    def predict(self, x):
        return torch.sigmoid(self.logit(x)).numpy()


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=1024,
          lr=2e-3, weight_decay=1e-5, verbose=True, seed=0):
    """Standard supervised training; returns the best validation accuracy."""
    torch.manual_seed(seed)
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.float32)
    xv = torch.as_tensor(x_val, dtype=torch.float32)
    yv = torch.as_tensor(y_val, dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(xt) // batch_size + 1)
    )
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(xt)
    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            out = model(xt[idx])
            loss = loss_fn(out, yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv))
            acc = float(((pv > 0.5).float() == yv).float().mean())
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  epoch {ep + 1:3d}/{epochs}  loss {tot / n:.5f}  val-acc {acc * 100:.3f}%")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return best_acc


def train_online(model, sample_fn, x_val, y_val, epochs=25, samples_per_epoch=200_000,
                 batch_size=512, lr=1e-3, weight_decay=1e-5, verbose=True, seed=0,
                 ema_decay=0.999, ema_from=0.5):
    """Train on freshly generated data every epoch, with weight averaging.

    The paper trains on 2**24 samples.  Holding that many t-bit vectors in RAM
    is wasteful when the generator is cheap, so instead we draw a fresh epoch
    each time; the network then never sees the same sample twice, which removes
    the overfitting a fixed 2*10**5-sample set shows on this very low-signal
    task.  ``sample_fn(n, rng) -> (X, Y)``.

    Weight averaging matters more here than it usually would.  The CRD does not
    use the network's *decision*, it sums the network's *logits*, so what the
    attack needs is a well-calibrated logit, not merely an accurate sign.  On a
    52%-accuracy task the gradient signal is tiny next to the gradient noise,
    so SGD leaves a substantial random component in the weights: the decision
    boundary lands in the right place (accuracy reaches the Bayes reference)
    while the logit picks up noise that is not a function of the sufficient
    statistic.  That noise is pure loss for the CRD.  An exponential moving
    average of the weights over the second half of training averages it away;
    measured on the 6-round distinguisher it lifts the fraction of logit
    variance explained by the popcount from about 0.68 to well above 0.9, and
    the multiple-bit attack's success rate with it.

    Selection between the raw and averaged weights is by validation accuracy.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 12345)
    xv = torch.as_tensor(x_val, dtype=torch.float32)
    yv = torch.as_tensor(y_val, dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, samples_per_epoch // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps_per_epoch + epochs
    )
    loss_fn = nn.BCEWithLogitsLoss()
    ema = torch.optim.swa_utils.AveragedModel(
        model, avg_fn=lambda a, c, n: ema_decay * a + (1.0 - ema_decay) * c)
    ema_start = int(epochs * ema_from)
    ema_ready = False

    def val_acc(m):
        m.eval()
        with torch.no_grad():
            return float(((torch.sigmoid(m(xv)) > 0.5).float() == yv).float().mean())

    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        x_np, y_np = sample_fn(samples_per_epoch, rng)
        xt = torch.as_tensor(x_np, dtype=torch.float32)
        yt = torch.as_tensor(y_np, dtype=torch.float32)
        model.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
            if ep >= ema_start:
                ema.update_parameters(model)
                ema_ready = True

        acc = val_acc(model)
        candidates = [(acc, model)]
        acc_ema = float("nan")
        if ema_ready:
            acc_ema = val_acc(ema.module)
            candidates.append((acc_ema, ema.module))
        for a, m in candidates:
            if a > best_acc:
                best_acc = a
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if verbose:
            extra = f"  ema-acc {acc_ema * 100:.3f}%" if ema_ready else ""
            print(f"  epoch {ep + 1:3d}/{epochs}  loss {tot / len(xt):.5f}  "
                  f"val-acc {acc * 100:.3f}%{extra}")
        del xt, yt, x_np, y_np
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return best_acc


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save(model, path, meta=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "dim": model.dim, "width": model.width, "depth": model.depth,
        "meta": meta or {},
    }, path)


def load(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = NeuralDistinguisher(blob["dim"], blob["width"], blob["depth"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob.get("meta", {})
