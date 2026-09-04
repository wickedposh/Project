"""
Does the world model actually use the action, or does it predict x_{t+1} from
the state alone?

This is Amir Bar's Q1 applied to this simulation. The action a_t (the aggregate
accept rate) is almost determined by the state: corr(x_t, a_t) ~ -0.91 across
seeds, so a model could achieve good prediction while ignoring a_t entirely.
About 40% of a_t's variation survives regressing on x_t, so there IS residual
signal -- the question is whether the model exploits it.

Three arms, identical architecture and training budget, differing only in the
action input:

  full     : (state history, a_t)          -- as in world_model_dim.py
  no-act   : (state history, 0)            -- action zeroed out
  shuffled : (state history, a_perm)       -- action present but scrambled
                                              across samples, so it carries
                                              no information about this row

The shuffled arm is the important control. If `full` matches `shuffled`, the
model is ignoring the action. If `full` beats both `no-act` and `shuffled`,
the action is contributing real signal beyond the state.
"""
import numpy as np
import torch
import torch.nn as nn

from world_model_dim import collect, windows, WorldModel


def fit_arm(X, A, Y, d, mode, epochs=120, bs=256, lr=1e-3, val_frac=0.2, seed=0):
    """mode in {'full', 'no-act', 'shuffled'}."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    A = A.copy()
    if mode == "no-act":
        A[:] = 0.0
    elif mode == "shuffled":
        A = A[rng.permutation(len(A))]
    elif mode != "full":
        raise ValueError(mode)

    n = len(X)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    vi, ti = perm[:n_val], perm[n_val:]

    xm, xs = X.mean(0), X.std(0) + 1e-8
    am, asd = A.mean(0), A.std(0) + 1e-8
    ym, ys = Y.mean(0), Y.std(0) + 1e-8
    Xn, An, Yn = (X - xm) / xs, (A - am) / asd, (Y - ym) / ys

    Xtr, Atr, Ytr = (torch.tensor(Xn[ti]), torch.tensor(An[ti]),
                     torch.tensor(Yn[ti]))
    Xva, Ava, Yva = (torch.tensor(Xn[vi]), torch.tensor(An[vi]),
                     torch.tensor(Yn[vi]))

    m = WorldModel(X.shape[1], d)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.MSELoss()

    best = float("inf")
    for _ in range(epochs):
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            j = idx[i:i + bs]
            opt.zero_grad()
            lossf(m(Xtr[j], Atr[j]), Ytr[j]).backward()
            opt.step()
        with torch.no_grad():
            best = min(best, lossf(m(Xva, Ava), Yva).item())
    return best


if __name__ == "__main__":
    print("Collecting from the unaltered model...")
    data = collect(T=25000, seed=0)
    print(f"{len(data)} transitions")
    r_xa = np.corrcoef(data[:, 0], data[:, 2])[0, 1]
    b = np.polyfit(data[:, 0], data[:, 2], 1)
    resid = data[:, 2] - np.polyval(b, data[:, 0])
    print(f"corr(x, a) = {r_xa:+.4f};  "
          f"{100*resid.std()/data[:,2].std():.0f}% of the action's variation "
          f"survives regressing on x\n")

    L, d = 5, 1          # the setting the L x d sweep settled on
    X, A, Y = windows(data, L)
    seeds = (0, 1, 2)

    print(f"Ablation at L={L}, d={d}, {len(seeds)} seeds:\n")
    res = {}
    for mode in ("full", "no-act", "shuffled"):
        errs = np.array([fit_arm(X, A, Y, d, mode, seed=s) for s in seeds])
        res[mode] = errs
        print(f"  {mode:>9}: val MSE {errs.mean():.5f} +/- {errs.std(ddof=1):.5f}")

    print()
    for ctrl in ("no-act", "shuffled"):
        diff = res[ctrl] - res["full"]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"  {ctrl} minus full: {diff.mean():+.5f}  SE {se:.5f}  "
              f"95% CI [{diff.mean()-1.96*se:+.5f}, {diff.mean()+1.96*se:+.5f}]")

    print("\nReading: if the controls are no worse than `full`, the model is")
    print("ignoring the action -- the state alone carries the prediction.")
    print("If `full` is clearly better, the action's residual variation is")
    print("being used despite corr(x, a) ~ -0.91.")
