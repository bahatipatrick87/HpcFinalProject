"""
Amdahl's Law analysis of the measured strong scaling (report Section 5.3.3,
Table 5.2, Figures 5.2 and 5.3).

Amdahl's Law: S_P = 1 / (f + (1 - f) / P), with ceiling S_inf = 1 / f.
The effective non-scaling fraction f is fitted from the measured P=8
speedup, f = (P / S_P - 1) / (P - 1), then used to predict S_P at every
measured P and to bracket the ceiling at f +/- 0.06.

Reads results/strong_scaling.csv (written by run_scaling_experiments.py).

Usage:
    python experiments/analyze_amdahl.py

Output: results/amdahl_fit.csv
"""
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIT_P = 8
BRACKET = 0.06


def amdahl_speedup(f: float, p: float) -> float:
    return 1.0 / (f + (1.0 - f) / p)


def fit_f(p: int, speedup: float) -> float:
    """Solve Amdahl's Law for f from one measured (P, S_P) point."""
    return (p / speedup - 1.0) / (p - 1.0)


def karp_flatt(p: int, speedup: float) -> float:
    """Experimentally determined serial fraction e = (1/S - 1/P) / (1 - 1/P)."""
    return (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)


def measured_speedups():
    """{P: S_P} from the mean times in results/strong_scaling.csv."""
    with (RESULTS_DIR / "strong_scaling.csv").open() as f:
        rows = list(csv.DictReader(f))
    t = {int(r["ranks"]): float(r["elapsed_mean_s"]) for r in rows}
    return {p: t[1] / t[p] for p in sorted(t)}


def main():
    s = measured_speedups()
    f = fit_f(FIT_P, s[FIT_P])
    print(f"Fitted from P={FIT_P} (S={s[FIT_P]:.3f}): f = {f:.4f}, "
          f"ceiling 1/f = {1 / f:.2f}x")
    for fb in (f - BRACKET, f + BRACKET):
        print(f"  bracket f = {fb:.3f}: ceiling {1 / fb:.2f}x")

    rows = []
    print(f"\n{'P':>2} {'measured':>9} {'Amdahl':>7} {'diff':>7} {'Karp-Flatt':>11}")
    for p, sp in s.items():
        pred = amdahl_speedup(f, p)
        diff = 100 * (sp - pred) / pred
        e = karp_flatt(p, sp) if p > 1 else float("nan")
        rows.append((p, sp, pred, diff, e))
        print(f"{p:>2} {sp:>9.3f} {pred:>7.3f} {diff:>+6.1f}% {e:>11.4f}")

    with (RESULTS_DIR / "amdahl_fit.csv").open("w") as out:
        out.write("P,measured_speedup,amdahl_speedup,difference_pct,f,ceiling,karp_flatt\n")
        for p, sp, pred, diff, e in rows:
            out.write(f"{p},{sp:.4f},{pred:.4f},{diff:.2f},{f:.4f},{1 / f:.4f},{e:.4f}\n")
    print(f"\nWritten to {RESULTS_DIR / 'amdahl_fit.csv'}")


if __name__ == "__main__":
    main()
