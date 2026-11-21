"""Experiments for OOA: classic suite, CEC2017 subset, ablation, engineering
design, market-data calibration, and the report (tables + figures).

Usage:
  python3 experiments.py classic | cec | ablation | eng | calib | report | all
Raw runs are cached in results/runs_<stage>.pkl. Re-running a stage reuses the
cache; delete the file to recompute.
"""
import os
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import problems as P
from optimizers import ALGORITHMS, ABLATIONS, LEAN

warnings.filterwarnings("ignore")
OUT = "results"
os.makedirs(OUT, exist_ok=True)

N_POP = 30
REG = dict(ALGORITHMS)
REG.update(ABLATIONS)
REG.update(LEAN)
ALGS = list(ALGORITHMS)
FINAL = ["OOA-Lean", "OOA"] + [a for a in ALGS if a != "OOA"]

# Okabe-Ito based, checked with the palette validator (CVD and normal-vision
# separation pass). Markers and line styles repeat the identity redundantly.
STYLE = {
    "OOA-Lean": dict(color="#0072B2", ls="-", marker="o"),
    "OOA": dict(color="#0072B2", ls="-", marker="o"),
    "PSO": dict(color="#E69F00", ls="--", marker="s"),
    "GWO": dict(color="#009E73", ls="-.", marker="^"),
    "DE": dict(color="#D55E00", ls=":", marker="D"),
    "WOA": dict(color="#CC79A7", ls="--", marker="v"),
    "SSA": dict(color="#56B4E9", ls="-.", marker="P"),
    "CMA-ES": dict(color="#6A3D9A", ls="-", marker="X"),
}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Tinos", "DejaVu Serif"],
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ------------------------------------------------------------------ runner
def _job(args):
    prob, alg, seed, budget = args
    f, lb, ub = P.get_problem(prob)
    bx, bf, hist = REG[alg](f, lb, ub, n_pop=N_POP, budget=budget, seed=seed)
    return prob, alg, seed, float(bf), np.asarray(hist, np.float64), np.asarray(bx)


def run_stage(tag, probs, algs, seeds, budget):
    path = f"{OUT}/runs_{tag}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    jobs = [(p, a, s, budget) for p in probs for a in algs for s in seeds]
    print(f"[{tag}] {len(jobs)} runs", flush=True)
    runs = {(p, a): [] for p in probs for a in algs}
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for k, (p, a, s, bf, h, bx) in enumerate(ex.map(_job, jobs, chunksize=2)):
            runs[(p, a)].append((s, bf, h, bx))
            if (k + 1) % 100 == 0:
                print(f"[{tag}] {k + 1}/{len(jobs)}", flush=True)
    with open(path, "wb") as fh:
        pickle.dump(runs, fh)
    return runs


def load(tag):
    with open(f"{OUT}/runs_{tag}.pkl", "rb") as fh:
        return pickle.load(fh)


# ------------------------------------------------------------------ stats
def summarize(runs, probs, algs):
    rows = []
    for p in probs:
        for a in algs:
            v = np.array([r[1] for r in runs[(p, a)]])
            rows.append(dict(problem=p.split(":")[1], algorithm=a, runs=v.size,
                             mean=v.mean(), std=v.std(ddof=1), median=np.median(v),
                             best=v.min(), worst=v.max()))
    return pd.DataFrame(rows)


def rank_table(summary, algs):
    med = summary.pivot(index="problem", columns="algorithm", values="median")[algs]
    ranks = med.rank(axis=1, method="average")
    return med, ranks


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def friedman_vs_control(ranks, control):
    """Friedman test plus Holm-adjusted z-tests of each algorithm vs control."""
    n, k = ranks.shape
    chi2, pf = stats.friedmanchisquare(*[ranks[c].to_numpy() for c in ranks.columns])
    avg = ranks.mean()
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    others = [c for c in ranks.columns if c != control]
    z = np.array([(avg[c] - avg[control]) / se for c in others])
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ph = holm(p)
    tab = pd.DataFrame({"algorithm": [control] + others,
                        "avg_rank": [avg[control]] + [avg[c] for c in others],
                        "z_vs_control": [np.nan] + list(z),
                        "p_raw": [np.nan] + list(p),
                        "p_holm": [np.nan] + list(ph)})
    return tab.sort_values("avg_rank").reset_index(drop=True), chi2, pf, n


def pairwise_vs_control(runs, probs, algs, control):
    """Per-problem rank-sum tests of control vs each algorithm."""
    rows = []
    for p in probs:
        c = np.array([r[1] for r in runs[(p, control)]])
        for a in algs:
            if a == control:
                continue
            v = np.array([r[1] for r in runs[(p, a)]])
            pv = stats.ranksums(c, v).pvalue
            verdict = "tie"
            if pv < 0.05:
                verdict = "better" if np.median(c) < np.median(v) else "worse"
            rows.append(dict(problem=p.split(":")[1], algorithm=a, p=pv, control_is=verdict))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ stages
def stage_classic():
    runs = run_stage("classic", P.CLASSIC_NAMES, ALGS, range(30), 30000)
    return runs


def stage_cec():
    return run_stage("cec", P.CEC_NAMES, ALGS, range(15), 30000)


def stage_ablation():
    return run_stage("ablation", P.CLASSIC_NAMES, list(ABLATIONS), range(20), 30000)


def stage_eng():
    return run_stage("eng", P.ENG_NAMES, ALGS, range(30), 15000)


# -------------------------------------------------------------- calibration
def load_market():
    df = pd.read_csv("market-data.csv", encoding="utf-8-sig")
    df["t"] = np.arange(len(df)) / 12.0  # rows are consecutive months
    return df


def make_model(df):
    ratio = (df["Demand"] / df["Supply"]).to_numpy()
    t = df["t"].to_numpy()
    y = df["Price"].to_numpy()

    def predict(theta):
        theta = np.atleast_2d(theta)
        a, b, c, d = (theta[:, k:k + 1] for k in range(4))
        return a * ratio[None, :] ** b * np.exp(c * t[None, :]) + d

    def rmse(theta):
        return np.sqrt(np.mean((predict(theta) - y[None, :]) ** 2, axis=1))

    return predict, rmse, y


CAL_LB = np.array([0.0, -15.0, -0.5, -6000.0])
CAL_UB = np.array([20000.0, 15.0, 0.5, 6000.0])


def stage_calib():
    path = f"{OUT}/runs_calib.pkl"
    df = load_market()
    predict, rmse, y = make_model(df)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh), df, predict, y
    res = {}
    for a in ALGS:
        res[a] = []
        for s in range(30):
            th, f, h = ALGORITHMS[a](rmse, CAL_LB, CAL_UB, n_pop=N_POP, budget=20000, seed=s)
            res[a].append((s, float(f), h, th))
    with open(path, "wb") as fh:
        pickle.dump(res, fh)
    return res, df, predict, y


# ------------------------------------------------------------------ report
def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=300)
    plt.close(fig)


def _line(ax, a, x, y, n_marks=7):
    lw = 2.0 if a.startswith("OOA") else 1.2
    z = 5 if a.startswith("OOA") else 3
    ax.plot(x, y, label=a, lw=lw, zorder=z, markevery=max(1, len(x) // n_marks),
            ms=4, mfc="white" if not a.startswith("OOA") else STYLE[a]["color"],
            **STYLE[a])


def convergence_figure(runs, probs, name, ncols, floor=1e-300, algs=None):
    algs = algs or ALGS
    nrows = int(np.ceil(len(probs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.1 * nrows + 0.5),
                             constrained_layout=True, squeeze=False)
    for ax, p in zip(axes.ravel(), probs):
        for a in algs:
            H = np.array([r[2] for r in runs[(p, a)]])
            y = np.log10(np.maximum(np.median(H, axis=0), floor))
            x = np.arange(1, y.size + 1) * N_POP
            _line(ax, a, x, y)
        ax.set_title(p.split(":")[1])
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel(r"$\log_{10}$ median best")
    for ax in axes.ravel()[len(probs):]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(algs), frameon=False)
    _save(fig, name)


def rank_bar(avg, name, title, colors=None):
    avg = avg.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(3.4, 0.38 * len(avg) + 0.7), constrained_layout=True)
    cols = [colors[a] if colors else "#0072B2" for a in avg.index]
    ax.barh(avg.index, avg.values, color=cols, height=0.62)
    for i, v in enumerate(avg.values):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title(title)
    ax.set_xlim(0, avg.max() * 1.15)
    ax.grid(axis="y", visible=False)
    _save(fig, name)


def report():
    pd.set_option("display.width", 220, "display.max_columns", 40)
    out = {}

    # classic
    rc = load("classic")
    sc = summarize(rc, P.CLASSIC_NAMES, ALGS)
    sc.to_csv(f"{OUT}/classic_summary.csv", index=False)
    pairwise_vs_control(rc, P.CLASSIC_NAMES, ALGS, "OOA").to_csv(
        f"{OUT}/classic_pairwise_vs_OOA.csv", index=False)
    convergence_figure(rc, P.CLASSIC_NAMES, "classic_convergence", 3)

    # cec
    rcec = load("cec")
    scec = summarize(rcec, P.CEC_NAMES, ALGS)
    scec.to_csv(f"{OUT}/cec2017_summary.csv", index=False)
    pairwise_vs_control(rcec, P.CEC_NAMES, ALGS, "OOA").to_csv(
        f"{OUT}/cec2017_pairwise_vs_OOA.csv", index=False)
    convergence_figure(rcec, [f"cec:F{k}" for k in (1, 4, 10, 12)], "cec2017_convergence", 2, floor=1e-12)

    # engineering
    re_ = load("eng")
    rows = []
    for p in P.ENG_NAMES:
        name = p.split(":")[1]
        best_known = P.ENG[name][4]
        for a in ALGS:
            raw = [P.eng_raw(name, r[3]) for r in re_[(p, a)]]
            feas = np.array([v <= 1e-4 for _, v in raw])
            fv = np.array([f for f, _ in raw])
            ff = fv[feas]
            rows.append(dict(problem=name, algorithm=a, feasible=f"{feas.sum()}/{len(raw)}",
                             best=ff.min() if ff.size else np.nan,
                             mean=ff.mean() if ff.size else np.nan,
                             std=ff.std(ddof=1) if ff.size > 1 else np.nan,
                             worst=ff.max() if ff.size else np.nan,
                             gap_best_pct=100 * (ff.min() - best_known) / best_known if ff.size else np.nan))
    seng = pd.DataFrame(rows)
    seng.to_csv(f"{OUT}/engineering_summary.csv", index=False)
    # ranks for engineering use penalized medians
    seng_pen = summarize(re_, P.ENG_NAMES, ALGS)

    # pooled Friedman across classic + CEC + engineering
    pooled = pd.concat([sc, scec, seng_pen])
    med, ranks = rank_table(pooled, ALGS)
    tab, chi2, pf, n = friedman_vs_control(ranks, "OOA")
    tab.to_csv(f"{OUT}/friedman_pooled.csv", index=False)
    ranks.to_csv(f"{OUT}/ranks_per_problem.csv")
    rank_bar(ranks.mean(), "friedman_ranks", f"Average rank over {n} problems",
             colors={a: STYLE[a]["color"] for a in ALGS})
    # per-suite Friedman average ranks for context
    suite_ranks = pd.DataFrame({
        "classic": rank_table(sc, ALGS)[1].mean(),
        "CEC2017": rank_table(scec, ALGS)[1].mean(),
        "engineering": rank_table(seng_pen, ALGS)[1].mean(),
        "pooled": ranks.mean()})
    suite_ranks.to_csv(f"{OUT}/ranks_by_suite.csv")
    print("\n== Average rank by suite (lower is better)")
    print(suite_ranks.round(2))
    print(f"\n== Pooled Friedman: chi2={chi2:.2f}, p={pf:.3g}, N={n} problems, k={len(ALGS)}")
    print(tab.round(4).to_string(index=False))

    # ablation
    ra = load("ablation")
    sa = summarize(ra, P.CLASSIC_NAMES, list(ABLATIONS))
    sa.to_csv(f"{OUT}/ablation_summary.csv", index=False)
    meda, ranksa = rank_table(sa, list(ABLATIONS))
    pw = pairwise_vs_control(ra, P.CLASSIC_NAMES, list(ABLATIONS), "OOA (full)")
    tally = pw.groupby("algorithm")["control_is"].value_counts().unstack(fill_value=0)
    tally = tally.reindex(columns=["better", "tie", "worse"], fill_value=0)
    tally.columns = ["full_better", "tie", "full_worse"]
    tally["avg_rank"] = ranksa.mean()[tally.index]
    tally.loc["OOA (full)"] = [np.nan, np.nan, np.nan, ranksa.mean()["OOA (full)"]]
    tally = tally.sort_values("avg_rank")
    tally.to_csv(f"{OUT}/ablation_ranks.csv")
    rank_bar(ranksa.mean(), "ablation_ranks", "OOA ablation, 6 functions")
    print("\n== Ablation (full vs variant, per-function rank-sum tests over 20 seeds)")
    print(tally.round(2))

    print("\n== Classic suite (median / mean)")
    print(sc.pivot(index="problem", columns="algorithm", values="median")[ALGS].map(lambda v: f"{v:.3g}"))
    print("\n== CEC2017 subset, error to optimum (median)")
    print(scec.pivot(index="problem", columns="algorithm", values="median")[ALGS].map(lambda v: f"{v:.3g}"))
    print("\n== Engineering (best feasible / feasible runs)")
    t = seng.assign(cell=lambda d: d.apply(lambda r: f"{r.best:.6g} ({r.feasible})", axis=1))
    print(t.pivot(index="problem", columns="algorithm", values="cell")[ALGS])

    # calibration
    res, df, predict, y = stage_calib()
    sst = np.sum((y - y.mean()) ** 2)
    rng = np.random.default_rng(0)
    ref_best, ref_theta = np.inf, None
    for _ in range(200):
        x0 = CAL_LB + rng.random(4) * (CAL_UB - CAL_LB)
        try:
            r = least_squares(lambda th: predict(th)[0] - y, x0, bounds=(CAL_LB, CAL_UB))
        except Exception:
            continue
        v = float(np.sqrt(np.mean(r.fun ** 2)))
        if v < ref_best:
            ref_best, ref_theta = v, r.x
    rows, best_theta = [], {}
    for a in ALGS:
        v = np.array([r[1] for r in res[a]])
        k = int(np.argmin(v))
        th = res[a][k][3]
        best_theta[a] = th
        sse = np.sum((predict(th)[0] - y) ** 2)
        rows.append(dict(method=a, rmse_best=v.min(), rmse_median=np.median(v),
                         rmse_mean=v.mean(), rmse_std=v.std(ddof=1), r2_best=1 - sse / sst,
                         a=th[0], b=th[1], c=th[2], d=th[3]))
    sse = np.sum((predict(ref_theta)[0] - y) ** 2)
    rows.append(dict(method="Multi-start LSQ (200 starts)", rmse_best=ref_best,
                     r2_best=1 - sse / sst, a=ref_theta[0], b=ref_theta[1],
                     c=ref_theta[2], d=ref_theta[3]))
    rows.append(dict(method="Mean-only baseline", rmse_best=float(np.std(y)), r2_best=0.0))
    cal = pd.DataFrame(rows)
    cal.to_csv(f"{OUT}/calibration_summary.csv", index=False)
    print("\n== Market calibration")
    print(cal.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    ax = axes[0]
    for a in ALGS:
        H = np.array([r[2] for r in res[a]])
        yv = np.median(H, axis=0)
        _line(ax, a, np.arange(1, yv.size + 1) * N_POP, yv)
    ax.axhline(ref_best, color="0.2", lw=0.8, ls=":", label="LSQ reference")
    ax.set_ylim(ref_best - 15, ref_best + 120)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Median best RMSE")
    ax.set_title("Calibration convergence")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax = axes[1]
    ax.plot(df["t"] + 2014, y, color="0.55", lw=1.0, label="Observed")
    ax.plot(df["t"] + 2014, predict(best_theta["OOA"])[0], color=STYLE["OOA"]["color"],
            lw=1.6, label="OOA fit")
    ax.set_xlabel("Year")
    ax.set_ylabel("Price")
    ax.set_title("Best OOA fit")
    ax.legend(frameon=False)
    _save(fig, "calibration")


# ------------------------------------------------------- confirmatory run
FSEEDS = 1000  # fresh seeds, disjoint from tuning (100+) and exploration (0 to 29)


def stage_fclassic():
    return run_stage("final_classic", P.CLASSIC_NAMES, FINAL, range(FSEEDS, FSEEDS + 30), 30000)


def stage_fcec():
    return run_stage("final_cec", P.CEC_NAMES, FINAL, range(FSEEDS, FSEEDS + 15), 30000)


def stage_feng():
    return run_stage("final_eng", P.ENG_NAMES, FINAL, range(FSEEDS, FSEEDS + 30), 15000)


def report_final():
    pd.set_option("display.width", 220, "display.max_columns", 40)
    D = f"{OUT}/final"
    os.makedirs(D, exist_ok=True)
    rc, rcec, re_ = load("final_classic"), load("final_cec"), load("final_eng")
    sc = summarize(rc, P.CLASSIC_NAMES, FINAL)
    scec = summarize(rcec, P.CEC_NAMES, FINAL)
    sc.to_csv(f"{D}/classic_summary.csv", index=False)
    scec.to_csv(f"{D}/cec2017_summary.csv", index=False)
    pairwise_vs_control(rc, P.CLASSIC_NAMES, FINAL, "OOA-Lean").to_csv(f"{D}/classic_pairwise.csv", index=False)
    pairwise_vs_control(rcec, P.CEC_NAMES, FINAL, "OOA-Lean").to_csv(f"{D}/cec2017_pairwise.csv", index=False)

    rows = []
    for p in P.ENG_NAMES:
        name = p.split(":")[1]
        best_known = P.ENG[name][4]
        for a in FINAL:
            raw = [P.eng_raw(name, r[3]) for r in re_[(p, a)]]
            feas = np.array([v <= 1e-4 for _, v in raw])
            fv = np.array([f for f, _ in raw])
            ff = fv[feas]
            rows.append(dict(problem=name, algorithm=a, feasible=int(feas.sum()), runs=len(raw),
                             best=ff.min() if ff.size else np.nan,
                             mean=ff.mean() if ff.size else np.nan,
                             std=ff.std(ddof=1) if ff.size > 1 else np.nan,
                             worst=ff.max() if ff.size else np.nan,
                             gap_best_pct=100 * (ff.min() - best_known) / best_known if ff.size else np.nan))
    seng = pd.DataFrame(rows)
    seng.to_csv(f"{D}/engineering_summary.csv", index=False)
    seng_pen = summarize(re_, P.ENG_NAMES, FINAL)
    pairwise_vs_control(re_, P.ENG_NAMES, FINAL, "OOA-Lean").to_csv(f"{D}/engineering_pairwise.csv", index=False)

    pooled = pd.concat([sc, scec, seng_pen])
    med, ranks = rank_table(pooled, FINAL)
    tab, chi2, pf, n = friedman_vs_control(ranks, "OOA-Lean")
    tab.to_csv(f"{D}/friedman_pooled.csv", index=False)
    ranks.to_csv(f"{D}/ranks_per_problem.csv")
    suite = pd.DataFrame({
        "classic": rank_table(sc, FINAL)[1].mean(),
        "CEC2017": rank_table(scec, FINAL)[1].mean(),
        "engineering": rank_table(seng_pen, FINAL)[1].mean(),
        "pooled": ranks.mean()})
    suite.to_csv(f"{D}/ranks_by_suite.csv")
    with open(f"{D}/friedman_stats.txt", "w") as fh:
        fh.write(f"chi2={chi2:.4f}\np={pf:.6g}\nN={n}\nk={len(FINAL)}\n")

    convergence_figure(rc, P.CLASSIC_NAMES, "final/classic_convergence", 3,
                       algs=["OOA-Lean"] + [a for a in ALGS if a != "OOA"])
    convergence_figure(rcec, [f"cec:F{k}" for k in (1, 4, 10, 12)], "final/cec2017_convergence", 2,
                       floor=1e-12, algs=["OOA-Lean"] + [a for a in ALGS if a != "OOA"])
    cols = {a: STYLE[a]["color"] for a in FINAL}
    cols["OOA"] = "#8FB8D8"
    rank_bar(ranks.mean(), "final/friedman_ranks", f"Average rank over {n} problems", colors=cols)

    print(f"== Pooled Friedman: chi2={chi2:.2f}, p={pf:.3g}, N={n}, k={len(FINAL)}")
    print(tab.round(4).to_string(index=False))
    print("\n== Average rank by suite")
    print(suite.round(2))
    print("\n== Classic median")
    print(sc.pivot(index="problem", columns="algorithm", values="median")[FINAL].map(lambda v: f"{v:.3g}"))
    print("\n== CEC2017 median error")
    print(scec.pivot(index="problem", columns="algorithm", values="median")[FINAL].map(lambda v: f"{v:.3g}"))
    t = seng.assign(cell=lambda d: d.apply(lambda r: f"{r.best:.6g} ({r.feasible}/{r.runs})", axis=1))
    print("\n== Engineering best feasible")
    print(t.pivot(index="problem", columns="algorithm", values="cell")[FINAL])


STAGES = {"classic": stage_classic, "cec": stage_cec, "ablation": stage_ablation,
          "eng": stage_eng, "calib": stage_calib,
          "fclassic": stage_fclassic, "fcec": stage_fcec, "feng": stage_feng}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "report":
        report()
    elif arg == "freport":
        report_final()
    elif arg == "all":
        for fn in STAGES.values():
            fn()
        report()
    else:
        STAGES[arg]()
        print(f"stage {arg} done")
