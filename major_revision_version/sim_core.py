"""Shared continuous-time model for Sections 2–3; store edges in arrays without clipping integrated states."""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.special import expit


def signed_power(z, alpha):
    z = np.asarray(z)
    return np.sign(z) * np.abs(z) ** alpha


def _active(active, n):
    out = np.asarray(sorted(set(active)), dtype=int)
    if out.size and (out.min() < 0 or out.max() >= n):
        raise ValueError('Active node index is outside the graph.')
    return out


def _gains(gains, active, n):
    """Accept a uniform gain, per-node gains, or gains ordered by active."""
    g = np.asarray(gains, dtype=float)
    if g.ndim == 0:
        out = np.full(len(active), float(g))
    elif g.shape == (n,):
        out = g[active]
    elif g.shape == (len(active),):
        out = g.copy()
    else:
        raise ValueError('Gains must be scalar, length N, or length len(active).')
    if not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError('Every prescribed active gain must be positive.')
    return out


class Model:
    def __init__(self, n, edges, A, epsilon=.5, beta=.8, alpha=.5,
                 mechanism='reciprocal', scale=1., normalize_strength=None):
        self.n = int(n)
        e = np.asarray(list(edges) if not isinstance(edges, np.ndarray) else edges,
                       dtype=int).reshape(-1, 2)
        e = np.sort(e, axis=1)
        if self.n < 2 or not len(e) or np.any(e < 0) or np.any(e >= self.n):
            raise ValueError('A graph with N >= 2 and valid edges is required.')
        if np.any(e[:, 0] == e[:, 1]) or len(np.unique(e, axis=0)) != len(e):
            raise ValueError('Use undirected edges once, without self loops.')
        self.edges = e[np.lexsort((e[:, 1], e[:, 0]))]
        self.i, self.j = self.edges.T
        self.degrees = np.bincount(self.edges.ravel(), minlength=self.n).astype(float)
        adjacency = coo_matrix((np.ones(2 * len(e)),
                               (np.r_[self.i, self.j], np.r_[self.j, self.i])),
                              shape=(self.n, self.n)).tocsr()
        if connected_components(adjacency, directed=False, return_labels=False) != 1:
            raise ValueError('The fixed undirected graph must be connected.')
        self.A = np.asarray(A, dtype=float).reshape(2, 2)
        self.epsilon, self.beta, self.alpha = map(float, (epsilon, beta, alpha))
        if not (0 < self.epsilon <= 1 and self.beta >= 0 and 0 < self.alpha < 1):
            raise ValueError('Require 0 < epsilon <= 1, beta >= 0, 0 < alpha < 1.')
        if mechanism not in ('reciprocal', 'unbiased', 'original'):
            raise ValueError('Unknown interaction mechanism.')
        self.mechanism, self.scale = mechanism, float(scale)
        if self.scale <= 0:
            raise ValueError('The weight scale must be positive.')
        self.normalize_strength = normalize_strength
        self.eta = (1 + self.alpha) / 2
        self._lower_laplacian = None
        self._eigen_cache = {}

    def payoff(self, x):
        x = np.asarray(x, dtype=float)
        neighbor_sum = (np.bincount(self.i, weights=x[self.j], minlength=self.n)
                        + np.bincount(self.j, weights=x[self.i], minlength=self.n))
        neighbor_mean = neighbor_sum / self.degrees
        a, b, c, d = self.A.ravel()
        return (a - b - c + d) * x * neighbor_mean + (b - d) * x + (c - d) * neighbor_mean + d

    def quantities(self, x):
        x = np.asarray(x, dtype=float)
        pi = self.payoff(x)
        q_ij = self.epsilon * expit(self.beta * (pi[self.j] - pi[self.i]))
        q_ji = self.epsilon * expit(self.beta * (pi[self.i] - pi[self.j]))
        if self.mechanism == 'reciprocal':
            w = np.minimum(q_ij / self.degrees[self.i], q_ji / self.degrees[self.j])
        elif self.mechanism == 'unbiased':
            w = self.epsilon / (2 * np.maximum(self.degrees[self.i], self.degrees[self.j]))
        else:
            # Effective weights from the original manuscript; this comparison model is not the new reciprocal matching rule.
            gap = np.abs((self.A[0, 1] - self.A[1, 0]) * (x[self.i] - x[self.j]))
            w = self.epsilon * expit(self.beta * gap) / (self.degrees[self.i] * self.degrees[self.j])
        w = self.scale * w
        if self.normalize_strength is not None:
            total = (self.normalize_strength(x) if callable(self.normalize_strength)
                     else self.normalize_strength)
            if total <= 0:
                raise ValueError('The target total directed edge weight must be positive.')
            w = w * (float(total) / (2 * w.sum()))
        return dict(pi=pi, q_ij=q_ij, q_ji=q_ji,
                    p_ij=self.degrees[self.i] * w,
                    p_ji=self.degrees[self.j] * w,
                    w_ij=w, w_ji=w.copy())

    def natural(self, x):
        x = np.asarray(x, dtype=float)
        w = self.quantities(x)['w_ij']
        flux = w * signed_power(x[self.j] - x[self.i], self.alpha)
        return (np.bincount(self.i, weights=flux, minlength=self.n)
                - np.bincount(self.j, weights=flux, minlength=self.n))

    def control(self, x, active, target, gains=1.):
        active = _active(active, self.n)
        u = np.zeros(self.n)
        u[active] = -_gains(gains, active, self.n) * signed_power(np.asarray(x)[active] - target, self.alpha)
        return u

    def rhs(self, x, active=(), target=.3, gains=1.):
        return self.natural(x) + self.control(x, active, target, gains)

    def value(self, x, target):
        return float(2 / (1 + self.alpha) * np.sum(np.abs(np.asarray(x) - target) ** (1 + self.alpha)))

    def state_cost(self, x, active, target, gains=1., f=None):
        active = _active(active, self.n)
        s = signed_power(np.asarray(x) - target, self.alpha)
        if f is None:
            f = self.natural(x)
        return float(-2 * s @ f + np.dot(_gains(gains, active, self.n), s[active] ** 2))

    def laplacian(self, weights):
        w = np.asarray(weights)
        rows = np.r_[self.i, self.j, self.i, self.j]
        cols = np.r_[self.i, self.j, self.j, self.i]
        return coo_matrix((np.r_[w, w, -w, -w], (rows, cols)),
                          shape=(self.n, self.n)).tocsr()

    def lower_laplacian(self):
        if self.normalize_strength is not None or self.mechanism == 'original':
            raise ValueError('The chapter 2 bound is not assigned to this comparison model.')
        if self._lower_laplacian is None:
            qmin = (self.epsilon / 2 if self.mechanism == 'unbiased'
                    else self.epsilon * expit(-self.beta * np.ptp(self.A)))
            lower = self.scale * qmin / np.maximum(self.degrees[self.i], self.degrees[self.j])
            self._lower_laplacian = self.laplacian(lower ** (1 / self.eta))
        return self._lower_laplacian

    def consensus_bound(self, x0):
        if 'lambda2' not in self._eigen_cache:
            lstar = self.lower_laplacian()
            if self.n <= 200:
                lam = np.linalg.eigvalsh(lstar.toarray())[1]
            else:
                lam = np.sort(eigsh(lstar, k=2, which='SM', tol=1e-10,
                                    v0=np.linspace(1., 2., self.n), return_eigenvectors=False))[1]
            self._eigen_cache['lambda2'] = float(lam)
        lam = self._eigen_cache['lambda2']
        e = np.asarray(x0) - np.mean(x0)
        return float((.5 * (e @ e)) ** (1 - self.eta) / (.5 * (4 * lam) ** self.eta * (1 - self.eta)))

    def pinned_eigenvalue(self, active, gains=1.):
        active = _active(active, self.n)
        if not len(active):
            return 0.
        k = _gains(gains, active, self.n)
        key = tuple(zip(active.tolist(), k.tolist()))
        if key not in self._eigen_cache:
            diagonal = np.zeros(self.n)
            diagonal[active] = k ** (1 / self.eta)
            h = self.lower_laplacian() + diags(diagonal)
            if self.n <= 200:
                lam = np.linalg.eigvalsh(h.toarray())[0]
            else:
                lam = eigsh(h, k=1, which='SM', tol=1e-10,
                            v0=np.linspace(1., 2., self.n), return_eigenvectors=False)[0]
            self._eigen_cache[key] = float(lam)
        return self._eigen_cache[key]

    def pinned_bound(self, x0, active, target, gains=1.):
        lam = self.pinned_eigenvalue(active, gains)
        if lam <= 0:
            return float('inf')
        e = np.asarray(x0) - target
        return float((.5 * (e @ e)) ** (1 - self.eta) / ((2 * lam) ** self.eta * (1 - self.eta)))


def simulate(model, x0, t_end, active=(), target=None, gains=1., controller=None,
             rtol=1e-7, atol=1e-9, max_step=.1, sample_dt=.1,
             arrival_tol=1e-3, stop_at_arrival=False, t_start=0., method='DOP853',
             error_fn=None):
    """Integrate a continuous interval with a fixed actuator set; retain the actual integrated endpoint.

    energy, mean_input, cost, and residual accumulate from t_start;
    mean_input = integral(sum(u)/N); residual is the squared remainder for the fixed inverse-optimal cost.
    controller(t, x, active) returns the actual N-dimensional input; extended controllers manage extra states.
    """
    x0 = np.asarray(x0, dtype=float).copy()
    if x0.shape != (model.n,) or np.any(x0 < 0) or np.any(x0 > 1):
        raise ValueError('Initial states must lie in [0,1]^N.')
    if t_end <= t_start or sample_dt <= 0 or arrival_tol <= 0:
        raise ValueError('Require t_end > t_start and positive sampling/arrival tolerance.')
    active = _active(active, model.n)
    k = _gains(gains, active, model.n)
    target = float(np.mean(x0) if target is None else target)
    if not 0 <= target <= 1:
        raise ValueError('The target must lie in [0,1].')
    error_fn = error_fn or (lambda x: np.max(np.abs(x - target)))

    def input_at(t, x):
        u = (model.control(x, active, target, gains) if controller is None
             else np.asarray(controller(t, x, active), dtype=float))
        if u.shape != (model.n,):
            raise ValueError('Controller must return an N-vector.')
        return u

    def rhs(t, y):
        x = y[:model.n]
        f = model.natural(x)
        u = input_at(t, x)
        s = signed_power(x - target, model.alpha)
        ustar = -k * s[active]
        ell = -2 * s @ f + np.dot(k, s[active] ** 2)
        weighted_energy = np.sum(u[active] ** 2 / k)
        remainder = np.sum((u[active] - ustar) ** 2 / k)
        return np.r_[f + u, u @ u, u.sum() / model.n,
                     ell + weighted_energy, remainder]

    def arrival_event(t, y):
        return float(error_fn(y[:model.n]) - arrival_tol)

    arrival_event.terminal = bool(stop_at_arrival)
    arrival_event.direction = -1
    y0 = np.r_[x0, np.zeros(4)]
    start_clock = perf_counter()
    if stop_at_arrival and error_fn(x0) <= arrival_tol:
        t = np.array([float(t_start)])
        y = y0[:, None]
        arrival, nfev = float(t_start), 0
    else:
        solution = solve_ivp(rhs, (t_start, t_end), y0, method=method,
                             rtol=rtol, atol=atol, max_step=max_step,
                             dense_output=True, events=arrival_event)
        if not solution.success:
            raise RuntimeError(solution.message)
        final_t = float(solution.t[-1])
        t = np.arange(t_start, final_t, sample_dt)
        if len(t) == 0 or not np.isclose(t[-1], final_t, atol=1e-12, rtol=0):
            t = np.r_[t, final_t]
        y = solution.sol(t)
        arrival = (float(t_start) if error_fn(x0) <= arrival_tol else
                   float(solution.t_events[0][0]) if len(solution.t_events[0]) else None)
        nfev = solution.nfev
    elapsed = perf_counter() - start_clock
    x = y[:model.n].T
    u = np.vstack([input_at(ti, xi) for ti, xi in zip(t, x)])
    error = np.array([error_fn(xi) for xi in x])
    balance = x.mean(axis=1) - x0.mean() - y[model.n + 1]
    return dict(t=t, x=x, u=u, energy=y[model.n], mean_input=y[model.n + 1],
                cost=y[model.n + 2], residual=y[model.n + 3], error=error,
                arrival=arrival, nfev=nfev, elapsed=elapsed, target=target,
                mean_balance=balance, state_min=float(x.min()), state_max=float(x.max()),
                input_max=float(np.abs(u).max()), active=active.copy(),
                method=method, rtol=rtol, atol=atol, max_step=max_step)


def replacement_metrics(model, x):
    q = model.quantities(x)
    w, pi = q['w_ij'], q['pi']
    intensity_edge = w * np.abs(pi[model.j] - pi[model.i])
    intensity = (np.bincount(model.i, weights=intensity_edge, minlength=model.n)
                 + np.bincount(model.j, weights=intensity_edge, minlength=model.n))
    weight_sum = (np.bincount(model.i, weights=w, minlength=model.n)
                  + np.bincount(model.j, weights=w, minlength=model.n))
    neighbor_sum = (np.bincount(model.i, weights=w * np.asarray(x)[model.j], minlength=model.n)
                    + np.bincount(model.j, weights=w * np.asarray(x)[model.i], minlength=model.n))
    return model.degrees.copy(), intensity, neighbor_sum / weight_sum


def replace_actuators(model, x, previous, responsive, delta=1., sigma=.1,
                      strategy='three-stage', rng=None):
    """Availability-only event: no state jumps; restart an empty set with only the lowest-index node."""
    if delta < 0 or sigma < 0:
        raise ValueError('Replacement tolerances must be nonnegative.')
    if strategy not in ('none', 'random', 'degree-neighbor', 'three-stage'):
        raise ValueError('Unknown replacement strategy.')
    previous, responsive = set(_active(previous, model.n)), set(_active(responsive, model.n))
    failed, active = previous - responsive, previous & responsive
    missed, chosen, branches = set(), [], []
    if strategy == 'none':
        return dict(active=sorted(active), missed=sorted(failed), chosen=chosen, branches=['none'])
    if strategy == 'random' and rng is None:
        raise ValueError('Random replacement requires an explicit seeded Generator.')
    degree, intensity, neighbor = replacement_metrics(model, x)
    mean_intensity = intensity.mean()
    for i in sorted(failed):
        available = responsive - active
        if not available:
            missed.add(i)
            branches.append('unavailable')
            continue
        if strategy == 'random':
            j = int(rng.choice(sorted(available)))
            branches.append('random')
        else:
            by_degree = {j for j in available if abs(degree[i] - degree[j]) <= delta}
            close = ({j for j in by_degree if abs(intensity[i] - intensity[j]) <= sigma * mean_intensity}
                     if strategy == 'three-stage' else set())
            pool = close or by_degree or available
            branches.append('full' if close else 'degree' if by_degree else 'available')
            j = min(pool, key=lambda j: (abs(neighbor[i] - neighbor[j]), j))
        active.add(j)
        chosen.append((int(i), int(j)))
    if not active and responsive:
        active.add(min(responsive))
        branches.append('restart')
    return dict(active=sorted(int(i) for i in active), missed=sorted(int(i) for i in missed),
                chosen=chosen, branches=branches)
