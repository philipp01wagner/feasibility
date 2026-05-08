"""
DeepJDOT (Damodaran et al., ECCV 2018), optimized for tabular data.

Performance fixes vs the naive version:
  1. Pure-PyTorch log-domain Sinkhorn (no CPU<->GPU roundtrip per batch,
     no POT call inside the training loop).
  2. Auto GPU detection.
  3. JDOT term computed every K batches (configurable), not every batch.
     The cheaper alignment-only Sinkhorn runs every batch.
  4. All data moved to device once (no DataLoader overhead — datasets are small).
  5. Larger default batch size to amortize Sinkhorn cost.
  6. Whole labeled-target set processed in one forward pass (fits easily at <100).
  7. Removed dependency on geomloss (one less moving part).
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_data
from features import standardize_and_normalize


# --------------------------- Models ---------------------------

class Encoder(nn.Module):
    def __init__(self, d_in, d_hidden=64, d_out=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.net = nn.Linear(d_in, n_classes)

    def forward(self, z):
        return self.net(z)


# --------------------------- Pure-Torch Sinkhorn ---------------------------

@torch.no_grad()
def sinkhorn_plan(M, reg=0.1, n_iters=30):
    """Entropic OT plan via log-domain Sinkhorn. Stable, pure-torch.

    Args:
        M: (ns, nt) cost matrix, normalized to roughly [0, 1].
        reg: entropic regularization.
        n_iters: number of Sinkhorn iterations.
    Returns:
        gamma: (ns, nt) transport plan, sums to 1.
    """
    ns, nt = M.shape
    device = M.device
    log_a = torch.full((ns,), -float(np.log(ns)), device=device)
    log_b = torch.full((nt,), -float(np.log(nt)), device=device)
    log_K = -M / reg
    log_u = torch.zeros(ns, device=device)
    log_v = torch.zeros(nt, device=device)
    for _ in range(n_iters):
        log_v = log_b - torch.logsumexp(log_K + log_u[:, None], dim=0)
        log_u = log_a - torch.logsumexp(log_K + log_v[None, :], dim=1)
    return (log_u[:, None] + log_K + log_v[None, :]).exp()


def sinkhorn_alignment(zs, zt, reg=0.1, n_iters=20):
    """Differentiable Sinkhorn-based alignment loss between two feature sets.

    Plan is detached (DeepJDOT trick); gradient flows through the cost so the
    encoder learns to bring the distributions together.
    """
    M = torch.cdist(zs, zt, p=2) ** 2
    M_norm = M / (M.detach().max() + 1e-8)
    gamma = sinkhorn_plan(M_norm.detach(), reg=reg, n_iters=n_iters)
    return (gamma * M).sum()


def jdot_loss(zs, ys, zt, logits_t, n_classes, reg=0.1, n_iters=20,
              alpha=1.0, beta=1.0):
    """Joint feature + label OT loss.

    cost(i,j) = alpha * ||z_s_i - z_t_j||^2 + beta * CE(y_s_i, softmax(logits_t_j))
    """
    M_feat = torch.cdist(zs, zt, p=2) ** 2
    log_pt = F.log_softmax(logits_t, dim=1)
    ys_oh = F.one_hot(ys, n_classes).float()
    M_lab = -ys_oh @ log_pt.t()

    M = alpha * M_feat + beta * M_lab
    M_norm = M / (M.detach().max() + 1e-8)
    gamma = sinkhorn_plan(M_norm.detach(), reg=reg, n_iters=n_iters)
    return (gamma * M).sum()


# --------------------------- Training ---------------------------

def run_deep_jdot(
    d_latent=32,
    epochs=80,
    batch_size=256,
    lr=1e-3,
    lambda_align=0.5,
    lambda_jdot=0.1,
    sinkhorn_reg=0.1,
    sinkhorn_iters=20,
    jdot_every=4,
    device=None,
    verbose=True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    data = load_data()
    Xs_list, Xt_lab, Xt_unl, Xt_test, _ = standardize_and_normalize(
        data["Xs_list"], data["Xt_train_labeled"],
        data["Xt_train_unlabeled"], data["Xt_test"],
    )
    ys_list = data["ys_list"]
    yt_lab = data["yt_train_labeled"]
    yt_test = data["yt_test"]
    n_classes = data["n_classes"]
    d = data["d"]

    X_src = np.vstack(Xs_list)
    y_src = np.concatenate(ys_list)
    Xt_for_align = np.vstack([Xt_unl, Xt_lab])

    # Move once. Tabular sizes fit trivially.
    X_src_t = torch.from_numpy(X_src).to(device)
    y_src_t = torch.from_numpy(y_src).to(device)
    Xt_align_t = torch.from_numpy(Xt_for_align).to(device)
    Xt_lab_t = torch.from_numpy(Xt_lab).to(device)
    yt_lab_t = torch.from_numpy(yt_lab).to(device)
    Xt_test_t = torch.from_numpy(Xt_test).to(device)

    n_src = len(X_src_t)
    n_tgt_align = len(Xt_align_t)

    enc = Encoder(d_in=d, d_out=d_latent).to(device)
    clf = Classifier(d_in=d_latent, n_classes=n_classes).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=lr)

    enc.train(); clf.train()
    batches_per_epoch = max(1, n_src // batch_size)

    for epoch in range(epochs):
        perm_src = torch.randperm(n_src, device=device)
        agg = {"ce_src": 0.0, "ce_tgt": 0.0, "align": 0.0, "jdot": 0.0, "n": 0}

        for b in range(batches_per_epoch):
            idx_s = perm_src[b * batch_size:(b + 1) * batch_size]
            xs = X_src_t[idx_s]
            ys = y_src_t[idx_s]
            idx_t = torch.randint(0, n_tgt_align, (batch_size,), device=device)
            xt = Xt_align_t[idx_t]

            zs = enc(xs)
            zt = enc(xt)
            zt_lab = enc(Xt_lab_t)

            logits_s = clf(zs)
            logits_t_lab = clf(zt_lab)

            ce_src = F.cross_entropy(logits_s, ys)
            ce_tgt = F.cross_entropy(logits_t_lab, yt_lab_t)
            align = sinkhorn_alignment(zs, zt, reg=sinkhorn_reg, n_iters=sinkhorn_iters)
            loss = ce_src + ce_tgt + lambda_align * align

            if b % jdot_every == 0:
                logits_t = clf(zt)
                j = jdot_loss(zs, ys, zt, logits_t, n_classes,
                              reg=sinkhorn_reg, n_iters=sinkhorn_iters)
                loss = loss + lambda_jdot * j
                agg["jdot"] += j.item() * len(xs)

            opt.zero_grad()
            loss.backward()
            opt.step()

            agg["ce_src"] += ce_src.item() * len(xs)
            agg["ce_tgt"] += ce_tgt.item() * len(xs)
            agg["align"] += align.item() * len(xs)
            agg["n"] += len(xs)

        if verbose and (epoch + 1) % 20 == 0:
            n = agg["n"]
            print(f"  epoch {epoch+1:3d}: ce_src={agg['ce_src']/n:.4f}  "
                  f"ce_tgt={agg['ce_tgt']/n:.4f}  align={agg['align']/n:.4f}  "
                  f"jdot={agg['jdot']/n:.4f}")

    enc.eval(); clf.eval()
    with torch.no_grad():
        preds = clf(enc(Xt_test_t)).argmax(dim=1).cpu().numpy()
    acc = float((preds == yt_test).mean())

    results = {"deep_jdot": acc}
    if verbose:
        print("=== DeepJDOT ===")
        print(f"  device: {device}")
        print(f"  acc:    {acc:.4f}")

    with open("results_deep_jdot.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run_deep_jdot()
