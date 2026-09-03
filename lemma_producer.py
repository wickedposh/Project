"""
Does a producer that reads ONE number off its belief do as well as one that
reads the whole curve?

Both arms run the same Bayesian belief update -- observe the accept rate,
update Ahat, exactly as in hetero_dynamic.py. They differ only in the read-out:

  full  : x_{t+1} = argmax_x (x - c) Ahat(x)        the whole curve
  lemma : x_{t+1} = c + 1/h~(x_t)                    one local number,
                                                     via the paper's Lemma

The belief mechanism is intact in both, so obfuscation still has its effect --
consumers corrupt the inference and that feeds through to future prices.

Each arm has its own consumer population, since each faces a different price
path and must best-respond to it. W draws are shared within a seed so the two
arms see identical underlying randomness.
"""
import numpy as np
import torch
import random

from hetero_dynamic import Game, ProducerBelief, SharedConsumer

HHAT_FLOOR = 0.005
ARMS = ["full", "lemma"]


def run(T=25000, beta=0.9, seed=0, N=300, log_every=500):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    game = Game(N=N)

    belief = {k: ProducerBelief(xbar=game.xbar) for k in ARMS}
    cons = {k: SharedConsumer(beta, game.xbar) for k in ARMS}
    price = {k: belief[k].best_price(game.cost) for k in ARMS}
    traj = {k: [] for k in ARMS}

    nu_init, nu_min, decay = 1.0, 0.05, 0.9998

    for t in range(T):
        nu = max(nu_min, nu_init * decay ** t)
        W = game.F_sampler(game.N)          # shared across arms

        for k in ARMS:
            b, c, x = belief[k], cons[k], price[k]

            cur = np.stack([W, np.full(game.N, x)], 1)
            a_c = c.act_batch(cur, nu)
            eps_i = c.eps_grid[a_c]

            p = game.accept_prob(W, x, eps_i)
            accept = (np.random.rand(game.N) < p).astype(float)
            acc_rate = accept.mean()
            profit = (x - game.cost) * acc_rate

            # --- the same Bayesian update in both arms ---
            b.update(x, acc_rate)

            # --- the read-out is what differs ---
            nxt = b.best_price(game.cost)

            if k == "full":
                h=b.hazard_at(nxt)
            else:
                h = 1/(nxt - game.cost)


            cons_r = (W - x) * accept
            idx = np.random.choice(game.N, size=min(32, game.N), replace=False)
            for i in idx:
                c.mem.push((W[i], x), int(a_c[i]), float(cons_r[i]), (W[i], nxt))
            c.learn(128)
            if t % 500 == 0:
                c.sync()

            if t % log_every == 0 and t > 0:
                traj[k].append({"t": t, "price": x, "profit": profit,
                                "eps": eps_i.mean(), "h": h})

            price[k] = nxt

    return traj


def summarise(all_traj, tail_frac=0.25):
    print(f"\n=== {len(all_traj)} seeds, final {int(tail_frac*100)}% of each run ===\n")
    stats = {}
    for k in ARMS:
        prof, pr, eps = [], [], []
        for tr in all_traj:
            rows = tr[k]
            tail = rows[int(len(rows) * (1 - tail_frac)):]
            prof.append(np.mean([r["profit"] for r in tail]))
            pr.append(np.mean([r["price"] for r in tail]))
            eps.append(np.mean([r["eps"] for r in tail]))
        stats[k] = (np.array(prof), np.array(pr), np.array(eps))

    for label, i in (("Producer profit", 0), ("Price", 1),
                     ("Aggregate obfuscation eps-bar", 2)):
        print(f"{label}:")
        for k in ARMS:
            v = stats[k][i]
            se = v.std(ddof=1) / np.sqrt(len(v))
            print(f"  {k:>6}: {v.mean():8.3f} +/- {v.std(ddof=1):6.3f}  SE {se:.3f}")
        print()

    d = stats["lemma"][0] - stats["full"][0]
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"Paired profit difference (lemma - full): {d.mean():+.3f}  SE {se:.3f}  "
          f"95% CI [{d.mean()-1.96*se:+.3f}, {d.mean()+1.96*se:+.3f}]")
    if d.mean() + 1.96 * se < 0:
        print("  -> reading the whole curve is significantly better;")
        print("     the single local number loses something.")
    elif d.mean() - 1.96 * se > 0:
        print("  -> the single number is significantly better. Unexpected.")
    else:
        print("  -> no significant difference: one number off the belief is")
        print("     as good as the whole curve, on this path.")


if __name__ == "__main__":
    seeds = list(range(8))
    all_traj = []
    for s in seeds:
        print(f"seed {s}...")
        all_traj.append(run(T=25000, beta=0.9, seed=s))
    summarise(all_traj)
