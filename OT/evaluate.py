"""Run all methods and produce a comparison table."""
import json
import os

from baselines import run_baselines
from ot_strategy_a import run_strategy_a
from ot_jcpot import run_jcpot
from deep_jdot import run_deep_jdot


def main():
    print("\n>>> Running baselines\n")
    r_base = run_baselines()
    print("\n>>> Running OT Strategy A\n")
    r_a = run_strategy_a()
    print("\n>>> Running JCPOT\n")
    r_jc = run_jcpot()
    print("\n>>> Running DeepJDOT\n")
    r_dj = run_deep_jdot()

    all_results = {}
    for d in (r_base, r_a, r_jc, r_dj):
        for k, v in d.items():
            if isinstance(v, float):
                all_results[k] = v

    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Markdown report
    sorted_items = sorted(all_results.items(), key=lambda kv: -kv[1])
    lines = ["# OT Transfer Learning Results\n",
             "| Method | Target Test Accuracy |",
             "|---|---|"]
    for k, v in sorted_items:
        lines.append(f"| {k} | {v:.4f} |")

    diag = []
    if "per_source_solo_acc" in r_a:
        diag.append("\n## Per-source solo accuracy (after OT, classifier on that source alone)\n")
        for i, a in enumerate(r_a["per_source_solo_acc"]):
            diag.append(f"- source {i}: {a:.4f}")
        diag.append("\n## Source weights (Strategy A, by inverse Wasserstein)\n")
        for i, w in enumerate(r_a["source_weights"]):
            diag.append(f"- source {i}: weight={w:.3f}  W(s,t)={r_a['wasserstein_distances'][i]:.4f}")

    report = "\n".join(lines) + "\n" + "\n".join(diag) + "\n"
    with open("report.md", "w") as f:
        f.write(report)

    print("\n" + report)


if __name__ == "__main__":
    main()
