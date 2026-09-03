"""
How much history does the world model need, and how far does it compress?

State   history of (x, h~) pairs over a window of length L
Action  a_t, the accept rate -- the observable realisation of the consumers'
        obfuscation choice, and all the producer actually sees
Target  x_{t+1}

Architecture, following the usual world-model shape: encode the state history
to z_t in R^d, encode the action to u_t, predict x_{t+1} from (z_t, u_t).

Sweeping both L and d because the two questions are entangled -- how much
history matters, and how far it compresses.

Note the transition is deterministic given enough history: the producer's
belief update is Ahat += lr*w*(a_t - Ahat) and then x_{t+1} = argmax, so
x_{t+1} is a function of (Ahat_t, x_t, a_t), and Ahat_t is built from the whole
past of (x, a). So prediction error should fall as L grows, and the d at which
it plateaus says how much of the 151-point belief curve actually matters for
the next price.

The game is hetero_dynamic.py, unaltered.
"""
import numpy as np
import torch
import torch.nn as nn
import random

from hetero_dynamic import Game, ProducerBelief, SharedConsumer

HHAT_FLOOR = 0.005


def collect(T=25000, beta=0.9, seed=0, N=300, burn_in=2000):
    """Log (x_t, h~_t, a_t, x_{t+1}) per step from the unaltered model."""
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

        h = belief.hazard_at(price)          # before the update, at x_t
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

        if t >= burn_in:
            rows.append((price, h, acc_rate, next_price))

        price = next_price

    return np.array(rows, dtype=np.float32)


def windows(data, L):
    """Build (state history of length L, action, target) tuples.
    Columns of `data`: x, h, a, x_next."""
    n = len(data)
    X, A, Y = [], [], []
    for t in range(L - 1, n):
        hist = data[t - L + 1:t + 1, :2]        # (L, 2) of (x, h)
        X.append(hist.reshape(-1))              # flattened, 2L features
        A.append(data[t, 2])                    # accept rate at t
        Y.append(data[t, 3])                    # x_{t+1}
    return (np.array(X, dtype=np.float32),
            np.array(A, dtype=np.float32).reshape(-1, 1),
            np.array(Y, dtype=np.float32).reshape(-1, 1))


class WorldModel(nn.Module):
    """Encoder: state history -> z in R^d. Action encoder: a -> u.
    Predictor: (z, u) -> x_{t+1}."""
    def __init__(self, n_state, d, hidden=64, u_dim=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_state, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d),
        )
        self.act_enc = nn.Sequential(
            nn.Linear(1, u_dim), nn.ReLU(),
        )
        self.pred = nn.Sequential(
            nn.Linear(d + u_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_hist, a):
        return self.pred(torch.cat([self.enc(x_hist), self.act_enc(a)], dim=1))


def fit(X, A, Y, d, epochs=120, bs=256, lr=1e-3, val_frac=0.2, seed=0):
    torch.manual_seed(seed)
    n = len(X)
    perm = np.random.RandomState(seed).permutation(n)
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
    for ep in range(epochs):
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
    print("Collecting (x, h~, a, x_next) from the unaltered model...")
    data = collect(T=25000, seed=0)
    print(f"{len(data)} transitions")
    print(f"  x : mean {data[:,0].mean():7.2f}  sd {data[:,0].std():.3f}")
    print(f"  h~: mean {data[:,1].mean():7.4f}  sd {data[:,1].std():.4f}")
    print(f"  a : mean {data[:,2].mean():7.3f}  sd {data[:,2].std():.3f}")

    Ls = (1, 2, 5, 10, 20)
    ds = (1, 2, 4, 8)
    seeds = (0, 1)

    print(f"\nValidation MSE (normalised; 1.0 = predicting the mean).")
    print(f"Sweeping window L and latent dimension d.\n")
    header = "  L \\ d " + "".join(f"{d:>12}" for d in ds)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for L in Ls:
        X, A, Y = windows(data, L)
        cells = []
        for d in ds:
            errs = np.array([fit(X, A, Y, d, seed=s) for s in seeds])
            cells.append(f"{errs.mean():.5f}")
        print(f"  {L:>3}   " + "".join(f"{c:>12}" for c in cells))

    print("\nReading down a column: how much history helps, at fixed capacity.")
    print("Reading across a row: how far that history compresses.")
    print("The (L, d) where the error stops falling is the answer.")
