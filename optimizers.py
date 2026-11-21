"""Optogenetic Optimization Algorithm (OOA) and baseline metaheuristics.

All optimizers minimize f(X) where X has shape (n, d) and f returns shape (n,).
All optimizers share one evaluation budget so comparisons are fair.
Each returns (best_x, best_f, history) where history[k] is the best-so-far
value after (k + 1) * n_pop evaluations.

Algorithms: OOA (proposed), PSO, GWO, DE (rand/1/bin), WOA, SSA, CMA-ES.
OOA ablation switches: adapt, use_expr, use_sched, burst, refractory, tau0.
"""
from functools import partial

import numpy as np


def _init(rng, n, lb, ub):
    return lb + rng.random((n, lb.size)) * (ub - lb)


def _fill(hist, evals, n_pop, best_f):
    """Append best-so-far for every completed block of n_pop evaluations."""
    while len(hist) * n_pop < evals:
        hist.append(best_f)


def _finish(hist, budget, n_pop, best_f):
    while len(hist) < budget // n_pop:
        hist.append(best_f)
    return np.array(hist[: budget // n_pop])


# ------------------------------------------------------------------ OOA
def ooa(f, lb, ub, n_pop=30, budget=30000, seed=0,
        tau0=0.05, refractory=1, p_floor=0.1, adapt=True,
        use_expr=True, use_sched=True, burst=False,
        stag_limit=30, burst_frac=0.3, elite_frac=0.2):
    """Optogenetic Optimization Algorithm.

    Neurons are candidate solutions. Each carries a per-dimension opsin
    expression level e_i. Three light operators act on active neurons:
      0 excitatory : pull toward the best neuron
      1 inhibitory : push away from a random neighbour
      2 random walk: small Gaussian pulse
    Operator probabilities adapt to recent success (wavelength selection).
    Intensity and duty cycle decay with progress. A refractory counter
    blocks updates after a successful spike. A spike threshold tau allows
    mildly worse moves early and shrinks to zero.
    Multi-source excitation: each excitatory pulse targets a random neuron
    from the top elite_frac of the population instead of only the best one,
    which slows premature collapse. elite_frac=0 uses the single best neuron.
    Optional inhibitory burst (off by default): when the best value stalls
    for stag_limit sweeps, a fraction of neurons is silenced and re-seeded
    by a strong inhibitory pulse around their old positions.
    Defaults were tuned on seeds 100 to 111, disjoint from benchmark seeds.
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    d = lb.size
    span = ub - lb

    X = _init(rng, n_pop, lb, ub)
    F = f(X)
    evals = n_pop
    expr = rng.uniform(0.3, 1.0, (n_pop, d)) if use_expr else np.ones((n_pop, d))
    refr = np.zeros(n_pop, int)
    succ = np.ones(3)
    tries = np.full(3, 2.0)
    stag = 0

    ib = int(np.argmin(F))
    best_x, best_f = X[ib].copy(), float(F[ib])
    hist = [best_f]

    while evals < budget:
        prog = evals / budget
        ps = prog if use_sched else 0.5
        # pulse schedule: strong and inhibitory early, weak and excitatory late
        s = 1.0 - 0.5 * ps
        i_exc = 0.4 + 1.1 * ps
        i_inh = 1.0 - 0.8 * ps
        duty = 0.9 - 0.4 * ps
        best_prev = best_f

        active = (refr == 0) & (rng.random(n_pop) < duty)
        refr[refr > 0] -= 1
        idx = np.flatnonzero(active)
        idx = idx[: max(0, budget - evals)]
        m = idx.size
        if m > 0:
            # wavelength selection
            p = succ / tries if adapt else np.ones(3)
            p = np.maximum(p / p.sum(), p_floor)
            p /= p.sum()
            w = np.array([0.5 + 0.5 * ps, 1.0 - 0.7 * ps, 0.6])
            p = p * w
            p /= p.sum()
            op = rng.choice(3, size=m, p=p)

            xi = X[idx]
            e = expr[idx]
            xj = X[rng.integers(0, n_pop, m)]
            r1 = rng.random((m, d))
            r2 = rng.random((m, d))
            # pulse noise shrinks cubically so late moves can refine
            noise = rng.standard_normal((m, d)) * span * 0.1 * (1.0 - prog) ** 3

            # excitatory light source: the best neuron, or a random elite one
            if elite_frac > 0:
                n_el = max(2, int(elite_frac * n_pop))
                elite = X[np.argsort(F)[:n_el]]
                tgt = elite[rng.integers(0, n_el, m)]
            else:
                tgt = np.broadcast_to(best_x, (m, d))

            step = np.zeros((m, d))
            k0, k1, k2 = op == 0, op == 1, op == 2
            step[k0] = i_exc * r1[k0] * (tgt[k0] - xi[k0]) \
                - 0.2 * i_inh * r2[k0] * (xj[k0] - xi[k0])
            step[k1] = 0.3 * i_exc * r1[k1] * (tgt[k1] - xi[k1]) \
                - i_inh * r2[k1] * (xj[k1] - xi[k1])
            delta = s * e * step + 0.05 * noise
            delta[k2] = e[k2] * noise[k2]
            Xn = np.clip(xi + delta, lb, ub)
            Fn = f(Xn)
            evals += m

            # spike-threshold acceptance
            tau = tau0 * (1.0 - prog) ** 3 * np.maximum(np.abs(F[idx]), 1e-12)
            ok = Fn < F[idx] + tau
            better = Fn < F[idx]

            for o in range(3):
                sel = op == o
                tries[o] += sel.sum()
                succ[o] += (sel & better).sum()

            acc = idx[ok]
            X[acc] = Xn[ok]
            F[acc] = Fn[ok]
            if refractory > 0:
                refr[idx[better]] = refractory
            if use_expr:
                moved = np.abs(Xn - xi) > 1e-12
                gain = better[:, None] & moved
                expr[idx] = np.clip(expr[idx] + 0.05 * gain
                                    - 0.01 * (~better[:, None]), 0.1, 1.0)

            ib = int(np.argmin(Fn))
            if Fn[ib] < best_f:
                best_f, best_x = float(Fn[ib]), Xn[ib].copy()

        # stagnation detection and inhibitory burst
        if best_f < best_prev - 1e-9 * max(abs(best_prev), 1e-12):
            stag = 0
        else:
            stag += 1
        if burst and stag >= stag_limit:
            k = min(max(1, int(burst_frac * n_pop)), budget - evals)
            if k > 0:
                order = np.argsort(F)
                pool = order[1:]  # keep the best neuron in place
                sel = rng.choice(pool, size=min(k, pool.size), replace=False)
                k = sel.size
                amp = span * (0.05 + 0.45 * (1.0 - prog))
                Xb = np.clip(X[sel] + i_inh * rng.uniform(-1, 1, (k, d)) * amp, lb, ub)
                Fb = f(Xb)
                evals += k
                X[sel], F[sel] = Xb, Fb
                if use_expr:
                    expr[sel] = rng.uniform(0.3, 1.0, (k, d))
                refr[sel] = 0
                ib = int(np.argmin(Fb))
                if Fb[ib] < best_f:
                    best_f, best_x = float(Fb[ib]), Xb[ib].copy()
            stag = 0

        _fill(hist, evals, n_pop, best_f)

    return best_x, best_f, _finish(hist, budget, n_pop, best_f)


# ------------------------------------------------------------------ PSO
def pso(f, lb, ub, n_pop=30, budget=30000, seed=0, c1=2.0, c2=2.0):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    d = lb.size
    span = ub - lb
    X = _init(rng, n_pop, lb, ub)
    V = rng.uniform(-0.1, 0.1, (n_pop, d)) * span
    F = f(X)
    evals = n_pop
    P, PF = X.copy(), F.copy()
    ig = int(np.argmin(PF))
    g, gf = P[ig].copy(), float(PF[ig])
    hist = [gf]
    vmax = 0.2 * span
    while evals + n_pop <= budget:
        w = 0.9 - 0.5 * evals / budget
        r1, r2 = rng.random((n_pop, d)), rng.random((n_pop, d))
        V = np.clip(w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X), -vmax, vmax)
        X = np.clip(X + V, lb, ub)
        F = f(X)
        evals += n_pop
        better = F < PF
        P[better], PF[better] = X[better], F[better]
        ig = int(np.argmin(PF))
        if PF[ig] < gf:
            gf, g = float(PF[ig]), P[ig].copy()
        hist.append(gf)
    return g, gf, _finish(hist, budget, n_pop, gf)


# ------------------------------------------------------------------ GWO
def gwo(f, lb, ub, n_pop=30, budget=30000, seed=0):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    d = lb.size
    X = _init(rng, n_pop, lb, ub)
    F = f(X)
    evals = n_pop
    order = np.argsort(F)
    A_, B_, D_ = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
    af = float(F[order[0]])
    hist = [af]
    while evals + n_pop <= budget:
        a = 2.0 * (1.0 - evals / budget)
        Xn = np.zeros_like(X)
        for leader in (A_, B_, D_):
            r1, r2 = rng.random((n_pop, d)), rng.random((n_pop, d))
            A = 2 * a * r1 - a
            C = 2 * r2
            Xn += leader - A * np.abs(C * leader - X)
        X = np.clip(Xn / 3.0, lb, ub)
        F = f(X)
        evals += n_pop
        order = np.argsort(F)
        if F[order[0]] < af:
            af = float(F[order[0]])
            A_ = X[order[0]].copy()
        B_, D_ = X[order[1]].copy(), X[order[2]].copy()
        hist.append(af)
    return A_, af, _finish(hist, budget, n_pop, af)


# ------------------------------------------------------------------- DE
def de(f, lb, ub, n_pop=30, budget=30000, seed=0, F_=0.5, CR=0.9):
    """Differential evolution, rand/1/bin."""
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    d = lb.size
    X = _init(rng, n_pop, lb, ub)
    F = f(X)
    evals = n_pop
    ib = int(np.argmin(F))
    bx, bf = X[ib].copy(), float(F[ib])
    hist = [bf]
    ar = np.arange(n_pop)
    while evals + n_pop <= budget:
        R = rng.random((n_pop, n_pop))
        R[ar, ar] = 2.0  # exclude self
        trio = np.argsort(R, axis=1)[:, :3]
        V = X[trio[:, 0]] + F_ * (X[trio[:, 1]] - X[trio[:, 2]])
        mask = rng.random((n_pop, d)) < CR
        mask[ar, rng.integers(0, d, n_pop)] = True
        U = np.clip(np.where(mask, V, X), lb, ub)
        FU = f(U)
        evals += n_pop
        better = FU <= F
        X[better], F[better] = U[better], FU[better]
        ib = int(np.argmin(F))
        if F[ib] < bf:
            bf, bx = float(F[ib]), X[ib].copy()
        hist.append(bf)
    return bx, bf, _finish(hist, budget, n_pop, bf)


# ------------------------------------------------------------------ WOA
def woa(f, lb, ub, n_pop=30, budget=30000, seed=0, b=1.0):
    """Whale optimization algorithm."""
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    d = lb.size
    X = _init(rng, n_pop, lb, ub)
    F = f(X)
    evals = n_pop
    ib = int(np.argmin(F))
    bx, bf = X[ib].copy(), float(F[ib])
    hist = [bf]
    while evals + n_pop <= budget:
        a = 2.0 * (1.0 - evals / budget)
        A = 2 * a * rng.random((n_pop, d)) - a
        C = 2 * rng.random((n_pop, d))
        p = rng.random((n_pop, 1))
        l = rng.uniform(-1, 1, (n_pop, 1))
        Xr = X[rng.integers(0, n_pop, n_pop)]
        enc = bx - A * np.abs(C * bx - X)
        exp = Xr - A * np.abs(C * Xr - X)
        spiral = np.abs(bx - X) * np.exp(b * l) * np.cos(2 * np.pi * l) + bx
        Xn = np.where(p < 0.5, np.where(np.abs(A) < 1, enc, exp), spiral)
        X = np.clip(Xn, lb, ub)
        F = f(X)
        evals += n_pop
        ib = int(np.argmin(F))
        if F[ib] < bf:
            bf, bx = float(F[ib]), X[ib].copy()
        hist.append(bf)
    return bx, bf, _finish(hist, budget, n_pop, bf)


# ------------------------------------------------------------------ SSA
def ssa(f, lb, ub, n_pop=30, budget=30000, seed=0):
    """Salp swarm algorithm."""
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    d = lb.size
    span = ub - lb
    X = _init(rng, n_pop, lb, ub)
    F = f(X)
    evals = n_pop
    ib = int(np.argmin(F))
    food, ff = X[ib].copy(), float(F[ib])
    hist = [ff]
    half = n_pop // 2
    while evals + n_pop <= budget:
        order = np.argsort(F)
        X = X[order]
        t = evals / budget
        c1 = 2.0 * np.exp(-(4.0 * t) ** 2)
        c2 = rng.random((half, d))
        c3 = rng.random((half, d))
        lead = np.where(c3 >= 0.5,
                        food + c1 * (span * c2 + lb),
                        food - c1 * (span * c2 + lb))
        Xn = X.copy()
        Xn[:half] = lead
        for i in range(half, n_pop):
            Xn[i] = 0.5 * (Xn[i] + Xn[i - 1])
        X = np.clip(Xn, lb, ub)
        F = f(X)
        evals += n_pop
        ib = int(np.argmin(F))
        if F[ib] < ff:
            ff, food = float(F[ib]), X[ib].copy()
        hist.append(ff)
    return food, ff, _finish(hist, budget, n_pop, ff)


# --------------------------------------------------------------- CMA-ES
def cmaes(f, lb, ub, n_pop=30, budget=30000, seed=0):
    """CMA-ES (no restarts) using the cma package with default population."""
    import cma
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    span = ub - lb
    # work in the unit cube so one step size suits every variable scale
    u0 = rng.random(lb.size)
    es = cma.CMAEvolutionStrategy(
        u0, 0.3,
        {"bounds": [0.0, 1.0], "seed": int(seed) + 1,
         "verbose": -9, "popsize": None, "maxfevals": budget})
    evals, bf, bx = 0, np.inf, lb + u0 * span
    hist = []
    while evals < budget and not es.stop():
        U = np.clip(np.array(es.ask()), 0.0, 1.0)
        U = U[: budget - evals]
        if U.shape[0] == 0:
            break
        fs = np.asarray(f(lb + U * span), float)
        evals += U.shape[0]
        j = int(np.argmin(fs))
        if fs[j] < bf:
            bf, bx = float(fs[j]), lb + U[j] * span
        if U.shape[0] == es.popsize:
            es.tell(U.tolist(), fs.tolist())
        _fill(hist, evals, n_pop, bf)
    if not hist:
        hist = [bf]
    return bx, bf, _finish(hist, budget, n_pop, bf)


ALGORITHMS = {"OOA": ooa, "PSO": pso, "GWO": gwo, "DE": de,
              "WOA": woa, "SSA": ssa, "CMA-ES": cmaes}

# Lean OOA: drops the two components the ablation did not support
# (refractory period, opsin expression). Chosen from the exploratory ablation
# on seeds 0 to 19, then confirmed on fresh seeds.
LEAN = {"OOA-Lean": partial(ooa, refractory=0, use_expr=False)}

# OOA ablation variants. "OOA" is the full algorithm.
ABLATIONS = {
    "OOA (full)": ooa,
    "+ inhibitory burst": partial(ooa, burst=True),
    "single light source": partial(ooa, elite_frac=0.0),
    "no refractory": partial(ooa, refractory=0),
    "no spike threshold": partial(ooa, tau0=0.0),
    "no adaptive wavelength": partial(ooa, adapt=False),
    "no opsin expression": partial(ooa, use_expr=False),
    "no pulse schedule": partial(ooa, use_sched=False),
}
