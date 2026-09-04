"""
Is the redundancy between price and hazard a property of the equilibrium, or
of the state space?

Everything measured so far discards the first few thousand steps as burn-in,
so all of it describes the equilibrium path. This measures the transient too --
the same run, no burn-in, correlations computed in windows across time.

The model is hetero_dynamic.py, unaltered. No perturbation, no frozen belief:
the producer visits off-equilibrium prices because it genuinely has not
converged yet, so the belief is informed at those prices and obfuscation still
has its effect.

If corr(x, h) is weak early and strong late, the Lemma x* = c + 1/h~* binds
only at the fixed point, and the one-dimensional structure found by the world
model is a property of the equilibrium rather than of the state space.
"""
import numpy as np
import torch
import random

from hetero_dynamic import Game, ProducerBelief, SharedConsumer

HHAT_FLOOR = 0.005


def collect_full(T=25000, beta=0.9, seed=0, N=300):
    """Log (t, x, h, a) from step 0 -- no burn-in."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    game = Game(N=N)
    belief = ProducerBelief(xbar=game.xbar)
    cons = SharedConsumer(beta, game.xbar)
    nu_init, nu_min, decay = 1.0, 0.05, 0.9998

    rows = []
    price = belief.best_price(game.cost)

    for t in range(T):
        nu = max(nu_min, nu_init * decay ** t)
        W = game.F_sampler(game.N)
        cur = np.stack([W, np.full(game.N, price)], 1)
        a_c = cons.act_batch(cur, nu)
        eps_i = cons.eps_grid[a_c]

        p = game.accept_prob(W, price, eps_i)
        accept = (np.random.rand(game.N) < p).astype(float)
        acc_rate = accept.mean()

        h = belief.hazard_at(price)
        belief.update(price, acc_rate)
        next_price = belief.best_price(game.cost)

        cons_r = (W - price) * accept
        idx = np.random.choice(game.N, size=min(32, game.N), replace=False)
        for i in idx:
            cons.mem.push((W[i], price), int(a_c[i]), float(cons_r[i]),
                          (W[i], next_price))
        cons.learn(128)
        if t % 500 == 0:
            cons.sync()

        rows.append((t, price, h, acc_rate, eps_i.mean()))
        price = next_price

    return np.array(rows, dtype=np.float32)


def windowed_correlation(data, n_windows=10):
    """corr(x, h) computed within successive time windows."""
    n = len(data)
    edges = np.linspace(0, n, n_windows + 1).astype(int)
    out = []
    for i in range(n_windows):
        lo, hi = edges[i], edges[i + 1]
        seg = data[lo:hi]
        seg = seg[seg[:, 2] >= HHAT_FLOOR]          # drop degenerate hazard
        if len(seg) < 20:
            out.append((lo, hi, np.nan, np.nan, np.nan, 0))
            continue
        x, h = seg[:, 1], seg[:, 2]
        r = np.corrcoef(x, h)[0, 1]
        out.append((lo, hi, r, x.std(), h.std(), len(seg)))
    return out


if __name__ == "__main__":
    seeds = (0, 1, 2)
    all_windows = []

    for s in seeds:
        print(f"seed {s}...")
        data = collect_full(T=25000, seed=s)
        all_windows.append(windowed_correlation(data))

    print(f"\n=== corr(x, h~) by time window, {len(seeds)} seeds ===\n")
    print("  window        corr (mean +/- sd)     sd(x)    sd(h~)")
    print("  " + "-" * 52)
    n_win = len(all_windows[0])
    for i in range(n_win):
        rs = np.array([w[i][2] for w in all_windows])
        sx = np.mean([w[i][3] for w in all_windows])
        sh = np.mean([w[i][4] for w in all_windows])
        lo, hi = all_windows[0][i][0], all_windows[0][i][1]
        if np.all(np.isnan(rs)):
            print(f"  {lo:>6}-{hi:<6}  (insufficient data)")
        else:
            print(f"  {lo:>6}-{hi:<6}  {np.nanmean(rs):+.4f} +/- {np.nanstd(rs):.4f}"
                  f"      {sx:6.3f}   {sh:.5f}")

    early = np.array([w[0][2] for w in all_windows])
    late = np.array([w[-1][2] for w in all_windows])
    print(f"\n  first window : {np.nanmean(early):+.4f}")
    print(f"  last window  : {np.nanmean(late):+.4f}")

    print("\nReading: if the correlation strengthens toward the end, the")
    print("redundancy is a property of the equilibrium, not the state space --")
    print("the Lemma binds at the fixed point and not before it.")
