"""Build LaTeX tables and number macros for the manuscript from result CSVs.

Reads results/final/*.csv (confirmatory fresh-seed run), results/ablation_*.csv
(exploratory ablation) and results/calibration_summary.csv.
Writes paper/tab_*.tex, paper/numbers.tex and copies figures to paper/fig/.
"""
import os
import shutil

import numpy as np
import pandas as pd

R = "results"
F = f"{R}/final"
OUT = "paper"
os.makedirs(f"{OUT}/fig", exist_ok=True)

ORDER = ["OOA-Lean", "OOA", "PSO", "GWO", "DE", "WOA", "SSA", "CMA-ES"]
CTRL = "OOA-Lean"
PRETTY = {"OOA-Lean": "OOA-L", "OOA": "OOA", "PSO": "PSO", "GWO": "GWO", "DE": "DE",
          "WOA": "WOA", "SSA": "SSA", "CMA-ES": "CMA-ES"}
MACRO = {"OOA-Lean": "LEAN", "OOA": "FULL", "PSO": "PSO", "GWO": "GWO", "DE": "DE",
         "WOA": "WOA", "SSA": "SSA", "CMA-ES": "CMAES"}


def num(v):
    """Compact LaTeX number with three significant digits."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if v == 0:
        return "0"
    a = abs(v)
    if 1e-2 <= a < 1e3:
        s = f"{v:.3g}"
        return s
    if 1e3 <= a < 1e4:
        return f"{v:.0f}"
    m, e = f"{v:.2e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def bold(s):
    """Bold a cell, including scientific-notation math cells."""
    if s.startswith("$"):
        return "$\\mathbf{" + s[1:-1] + "}$"
    return f"\\textbf{{{s}}}"


def pfmt(p):
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"\\ensuremath{{{m}{{\\times}}10^{{{int(e)}}}}}"
    return f"{p:.3f}"


def result_table(summary, pairwise, funcs, caption, label, stat="median", wide=True):
    piv = summary.pivot(index="problem", columns="algorithm", values=stat)[ORDER]
    piv = piv.loc[funcs]
    pw = pairwise.pivot(index="problem", columns="algorithm", values="control_is")
    cols = "l" + "r" * len(ORDER)
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             "Function & " + " & ".join(PRETTY[a] for a in ORDER) + " \\\\", "\\midrule"]
    for fn in funcs:
        row = piv.loc[fn]
        best = row.min()
        cells = []
        for a in ORDER:
            s = num(row[a])
            if row[a] == best:
                s = bold(s)
            cells.append(s)
        lines.append(fn.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    # win/tie/loss of the control against every other algorithm
    lines.append("\\midrule")
    wtl = []
    for a in ORDER:
        if a == CTRL:
            wtl.append("--")
            continue
        v = pw[a].loc[funcs]
        w, t, l = (v == "better").sum(), (v == "tie").sum(), (v == "worse").sum()
        wtl.append(f"{w}/{t}/{l}")
    lines.append("OOA-L w/t/l & " + " & ".join(wtl) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def eng_table(seng):
    probs = list(dict.fromkeys(seng["problem"]))
    cols = "ll" + "r" * len(ORDER)
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Engineering design problems. Best and mean feasible objective over 30 runs "
             "(feasibility tolerance $10^{-4}$). Bold marks the lowest value in each row.}",
             "\\label{tab:eng}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
             f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             "Problem & Stat & " + " & ".join(PRETTY[a] for a in ORDER) + " \\\\", "\\midrule"]
    names = {"PressureVessel": "Pressure vessel", "WeldedBeam": "Welded beam", "TensionSpring": "Tension spring"}
    for p in probs:
        d = seng[seng.problem == p].set_index("algorithm")
        for stat, lab in (("best", "best"), ("mean", "mean")):
            row = d[stat][ORDER]
            lo = row.min()
            cells = []
            for a in ORDER:
                v = row[a]
                s = f"{v:.6g}" if not np.isnan(v) else "--"
                if abs(v - lo) <= 1e-12 * max(1, abs(lo)):
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            first = names[p] if stat == "best" else ""
            lines.append(f"{first} & {lab} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if p != probs[-1] else "")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(l for l in lines if l != "") + "\n"


def rank_table():
    suite = pd.read_csv(f"{F}/ranks_by_suite.csv", index_col=0).loc[ORDER]
    fr = pd.read_csv(f"{F}/friedman_pooled.csv").set_index("algorithm")
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Average ranks (lower is better) by suite and pooled over all problems. "
             "$p_\\mathrm{Holm}$ tests each algorithm against OOA-L.}",
             "\\label{tab:rank}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Algorithm & Classic & CEC & Eng. & Pooled & $p_\\mathrm{Holm}$ \\\\", "\\midrule"]
    best = suite["pooled"].min()
    for a in sorted(ORDER, key=lambda x: suite.loc[x, "pooled"]):
        r = suite.loc[a]
        pooled = f"{r['pooled']:.2f}"
        if r["pooled"] == best:
            pooled = f"\\textbf{{{pooled}}}"
        ph = "--" if a == CTRL else pfmt(fr.loc[a, "p_holm"])
        lines.append(f"{PRETTY[a]} & {r['classic']:.2f} & {r['CEC2017']:.2f} & {r['engineering']:.2f} & {pooled} & {ph} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def ablation_table():
    ab = pd.read_csv(f"{R}/ablation_ranks.csv", index_col=0)
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Ablation of OOA on the six classic functions (20 seeds). Average rank "
             "(lower is better) and per-function rank-sum outcomes of the full algorithm against "
             "each variant (better/tie/worse).}",
             "\\label{tab:ablation}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lrr}", "\\toprule", "Variant & Avg. rank & Full b/t/w \\\\", "\\midrule"]
    for name, r in ab.sort_values("avg_rank").iterrows():
        if name == "OOA (full)":
            bt = "--"
        else:
            bt = f"{int(r['full_better'])}/{int(r['tie'])}/{int(r['full_worse'])}"
        lines.append(f"{name} & {r['avg_rank']:.2f} & {bt} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    sc = pd.read_csv(f"{F}/classic_summary.csv")
    scec = pd.read_csv(f"{F}/cec2017_summary.csv")
    pc = pd.read_csv(f"{F}/classic_pairwise.csv")
    pcec = pd.read_csv(f"{F}/cec2017_pairwise.csv")
    seng = pd.read_csv(f"{F}/engineering_summary.csv")

    classic_funcs = ["Sphere", "Rosenbrock", "Rastrigin", "Ackley", "Griewank", "Schwefel"]
    cec_funcs = ["F1", "F3", "F4", "F5", "F8", "F10", "F12", "F14", "F23"]
    open(f"{OUT}/tab_classic.tex", "w").write(result_table(
        sc, pc, classic_funcs,
        "Median final error over 30 runs on the 30-D classic functions (30\\,000 evaluations). "
        "Bold marks the lowest median. The last row counts functions where OOA-L is significantly "
        "better, tied, or worse (rank-sum test, $\\alpha=0.05$).", "tab:classic"))
    open(f"{OUT}/tab_cec.tex", "w").write(result_table(
        scec, pcec, cec_funcs,
        "Median error to the known optimum over 15 runs on nine CEC2017 functions (30-D, "
        "30\\,000 evaluations, opfunu numbering). Bold marks the lowest median.", "tab:cec"))
    open(f"{OUT}/tab_eng.tex", "w").write(eng_table(seng))
    open(f"{OUT}/tab_rank.tex", "w").write(rank_table())
    open(f"{OUT}/tab_ablation.tex", "w").write(ablation_table())

    # number macros
    suite = pd.read_csv(f"{F}/ranks_by_suite.csv", index_col=0)
    fr = pd.read_csv(f"{F}/friedman_pooled.csv").set_index("algorithm")
    st = dict(l.strip().split("=") for l in open(f"{F}/friedman_stats.txt"))
    cal = pd.read_csv(f"{R}/calibration_summary.csv").set_index("method")
    ab = pd.read_csv(f"{R}/ablation_ranks.csv", index_col=0)
    lines = ["% generated by make_tables.py; do not edit"]
    mac = lambda name, val: lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    mac("nProb", int(float(st["N"])))
    mac("nAlg", int(float(st["k"])))
    mac("friedChi", f"{float(st['chi2']):.1f}")
    mac("friedP", pfmt(float(st["p"])))
    for a in ORDER:
        mac(f"rank{MACRO[a]}", f"{suite.loc[a, 'pooled']:.2f}")
        for col, tag in (("classic", "Cl"), ("CEC2017", "Cec"), ("engineering", "Eng")):
            mac(f"rank{tag}{MACRO[a]}", f"{suite.loc[a, col]:.2f}")
        if a != CTRL:
            mac(f"pHolm{MACRO[a]}", pfmt(fr.loc[a, "p_holm"]))
    for r, name in (("OOA (full)", "Full"), ("+ inhibitory burst", "Burst"),
                    ("single light source", "Single"), ("no refractory", "NoRef"),
                    ("no spike threshold", "NoThr"), ("no adaptive wavelength", "NoAda"),
                    ("no opsin expression", "NoExp"), ("no pulse schedule", "NoSch")):
        mac(f"abl{name}", f"{ab.loc[r, 'avg_rank']:.2f}")
    eg = seng.set_index(["algorithm", "problem"])["gap_best_pct"]
    mac("gapLeanMax", f"{eg['OOA-Lean'].max():.3f}")
    mac("calRMSE", f"{cal.loc['OOA', 'rmse_best']:.2f}")
    mac("calRsq", f"{cal.loc['OOA', 'r2_best']:.3f}")
    mac("calBase", f"{cal.loc['Mean-only baseline', 'rmse_best']:.1f}")
    mac("calA", f"{cal.loc['OOA', 'a']:.0f}")
    mac("calB", f"{cal.loc['OOA', 'b']:.2f}")
    mac("calC", f"{cal.loc['OOA', 'c']:.4f}")
    mac("calD", f"{cal.loc['OOA', 'd']:.0f}")
    open(f"{OUT}/numbers.tex", "w").write("\n".join(lines) + "\n")

    for src, dst in (("final/classic_convergence", "classic_convergence"),
                     ("final/cec2017_convergence", "cec2017_convergence"),
                     ("final/friedman_ranks", "friedman_ranks"),
                     ("ablation_ranks", "ablation_ranks"),
                     ("calibration", "calibration")):
        shutil.copy(f"{R}/{src}.pdf", f"{OUT}/fig/{dst}.pdf")
    print("wrote tables, numbers.tex and figures to", OUT)


if __name__ == "__main__":
    main()
