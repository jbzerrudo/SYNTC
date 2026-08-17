"""
Two track propagators over the same 3-hourly increment target.

`StormBaseline`  reimplements the Bloemendaal et al. (2020) STORM approach:
                 inside each 5x5 degree cell, regress the current displacement
                 on the previous displacement, and sample the residual from a
                 Gaussian fitted to that cell.

`MDNPropagator`  replaces the per-cell regression with one neural network that
                 outputs a mixture of bivariate Gaussians over the displacement,
                 conditioned on position, motion, intensity, month and age.

Both expose the same two methods, `log_prob` and `sample`, so the benchmark can
score them with identical code and the comparison stays honest.  Crucially both
are *probability densities* over the same target, so their held-out log
likelihoods are directly comparable.
"""

import numpy as np
import torch
import torch.nn as nn

from data import FEATURES, TARGETS

LOG_2PI = float(np.log(2.0 * np.pi))

# No mixture component may be sharper than the position quantum. Without this
# a component can shrink onto a rounded coordinate and win likelihood for a
# measurement artefact rather than for a better description of storm motion.
MIN_LOG_SCALE = float(np.log(0.02))


# --------------------------------------------------------------------------
# STORM-style binned autoregression
# --------------------------------------------------------------------------
class StormBaseline:
    def __init__(self, cell_deg=5.0, min_count=25, ridge=1e-6):
        self.cell_deg = cell_deg
        self.min_count = min_count
        self.ridge = ridge
        self.cells = {}
        self.global_fit = None

    def _cell(self, lat, lon):
        return (
            np.floor(lat / self.cell_deg).astype(int),
            np.floor(lon / self.cell_deg).astype(int),
        )

    @staticmethod
    def _fit_block(X, Y, ridge):
        """Least squares Y ~ [1, X] plus a Gaussian on the residuals."""
        A = np.column_stack([np.ones(len(X)), X])
        G = A.T @ A + ridge * np.eye(A.shape[1])
        coef = np.linalg.solve(G, A.T @ Y)          # (3, 2)
        resid = Y - A @ coef
        cov = np.cov(resid.T) if len(resid) > 3 else np.eye(2) * 1e-3
        cov = np.atleast_2d(cov) + np.eye(2) * 1e-8
        return coef, cov

    def fit(self, frame):
        X = frame[["u_prev", "v_prev"]].to_numpy(float)
        Y = frame[TARGETS].to_numpy(float)
        self.global_fit = self._fit_block(X, Y, self.ridge)

        iy, ix = self._cell(frame["lat"].to_numpy(float), frame["lon"].to_numpy(float))
        keys = list(zip(iy.tolist(), ix.tolist()))
        order = {}
        for i, k in enumerate(keys):
            order.setdefault(k, []).append(i)

        for k, idx in order.items():
            if len(idx) >= self.min_count:
                idx = np.asarray(idx)
                self.cells[k] = self._fit_block(X[idx], Y[idx], self.ridge)
        return self

    def _params(self, frame):
        X = frame[["u_prev", "v_prev"]].to_numpy(float)
        iy, ix = self._cell(frame["lat"].to_numpy(float), frame["lon"].to_numpy(float))
        n = len(frame)
        mu = np.empty((n, 2))
        cov = np.empty((n, 2, 2))
        A = np.column_stack([np.ones(n), X])
        for i, k in enumerate(zip(iy.tolist(), ix.tolist())):
            coef, C = self.cells.get(k, self.global_fit)
            mu[i] = A[i] @ coef
            cov[i] = C
        return mu, cov

    def log_prob(self, frame):
        y = frame[TARGETS].to_numpy(float)
        mu, cov = self._params(frame)
        d = y - mu
        det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
        inv = np.empty_like(cov)
        inv[:, 0, 0] = cov[:, 1, 1] / det
        inv[:, 1, 1] = cov[:, 0, 0] / det
        inv[:, 0, 1] = -cov[:, 0, 1] / det
        inv[:, 1, 0] = -cov[:, 1, 0] / det
        q = np.einsum("ni,nij,nj->n", d, inv, d)
        return -0.5 * (q + np.log(det) + 2 * LOG_2PI)

    def sample(self, frame, rng, generator=None):
        mu, cov = self._params(frame)
        out = np.empty_like(mu)
        for i in range(len(mu)):
            out[i] = rng.multivariate_normal(mu[i], cov[i])
        return out


# --------------------------------------------------------------------------
# Mixture density network
# --------------------------------------------------------------------------
class MDN(nn.Module):
    def __init__(self, n_features, n_components=8, hidden=128):
        super().__init__()
        self.k = n_components
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # per component: 1 logit + 2 means + 2 log-scales + 1 correlation
        self.head = nn.Linear(hidden, n_components * 6)

    def forward(self, x):
        h = self.head(self.body(x))
        k = self.k
        logit, mu, log_s, rho_raw = torch.split(h, [k, 2 * k, 2 * k, k], dim=-1)
        mu = mu.view(-1, k, 2)
        # Scales are clamped well inside the observed increment range so a
        # component cannot collapse onto a single point and blow up the NLL.
        log_s = log_s.view(-1, k, 2).clamp(MIN_LOG_SCALE, 2.0)
        rho = torch.tanh(rho_raw) * 0.99
        return torch.log_softmax(logit, dim=-1), mu, log_s, rho

    @staticmethod
    def _component_log_prob(y, mu, log_s, rho):
        y = y.unsqueeze(1)                       # (n, 1, 2)
        z = (y - mu) * torch.exp(-log_s)         # standardised residuals
        z0, z1 = z[..., 0], z[..., 1]
        one_m = 1.0 - rho ** 2
        quad = (z0 ** 2 - 2 * rho * z0 * z1 + z1 ** 2) / one_m
        log_det = log_s.sum(-1) + 0.5 * torch.log(one_m)
        return -0.5 * quad - log_det - LOG_2PI

    def log_prob(self, x, y):
        log_w, mu, log_s, rho = self(x)
        return torch.logsumexp(log_w + self._component_log_prob(y, mu, log_s, rho), dim=-1)

    @torch.no_grad()
    def sample(self, x, generator=None):
        log_w, mu, log_s, rho = self(x)
        idx = torch.multinomial(log_w.exp(), 1, generator=generator).squeeze(-1)
        rows = torch.arange(len(x), device=x.device)
        m, s, r = mu[rows, idx], log_s[rows, idx].exp(), rho[rows, idx]
        e = torch.randn(len(x), 2, generator=generator, device=x.device)
        # Cholesky of the 2x2 correlation matrix, applied to standard normals.
        d0 = e[:, 0]
        d1 = r * e[:, 0] + torch.sqrt(1.0 - r ** 2) * e[:, 1]
        return m + torch.stack([d0, d1], dim=-1) * s


class MDNPropagator:
    """Wraps the network with the feature scaling and the training loop."""

    def __init__(self, n_components=8, hidden=128, seed=0):
        self.k = n_components
        self.hidden = hidden
        self.seed = seed
        self.net = None
        self.mean = None
        self.std = None

    def _x(self, frame):
        x = frame[FEATURES].to_numpy(np.float32)
        return (x - self.mean) / self.std

    def fit(self, frame, valid_frame=None, epochs=60, batch=512, lr=1e-3, verbose=True):
        torch.manual_seed(self.seed)
        raw = frame[FEATURES].to_numpy(np.float32)
        self.mean = raw.mean(0)
        self.std = raw.std(0)
        self.std[self.std < 1e-6] = 1.0

        X = torch.from_numpy(self._x(frame))
        Y = torch.from_numpy(frame[TARGETS].to_numpy(np.float32))
        self.net = MDN(X.shape[1], self.k, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        if valid_frame is not None:
            XV = torch.from_numpy(self._x(valid_frame))
            YV = torch.from_numpy(valid_frame[TARGETS].to_numpy(np.float32))

        n = len(X)
        best, best_state = -np.inf, None
        history = []
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, batch):
                j = perm[i:i + batch]
                opt.zero_grad()
                loss = -self.net.log_prob(X[j], Y[j]).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                total += float(loss) * len(j)
            sched.step()
            train_ll = -total / n

            if valid_frame is not None:
                self.net.eval()
                with torch.no_grad():
                    vll = float(self.net.log_prob(XV, YV).mean())
                history.append((ep, train_ll, vll))
                if vll > best:
                    best = vll
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                if verbose and (ep % 10 == 0 or ep == epochs - 1):
                    print(f"  epoch {ep:3d}  train ll {train_ll:8.4f}   valid ll {vll:8.4f}")
            else:
                history.append((ep, train_ll, np.nan))

        # Keep the parameters that were best on the validation years, not the
        # last ones, so the reported test number is not an overfit endpoint.
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.history = history
        return self

    def log_prob(self, frame):
        self.net.eval()
        with torch.no_grad():
            X = torch.from_numpy(self._x(frame))
            Y = torch.from_numpy(frame[TARGETS].to_numpy(np.float32))
            return self.net.log_prob(X, Y).numpy()

    def sample(self, frame, rng=None, generator=None):
        self.net.eval()
        X = torch.from_numpy(self._x(frame))
        return self.net.sample(X, generator=generator).numpy()
