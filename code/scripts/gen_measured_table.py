"""
gen_measured_table.py — read T_sweep_MUTAG.json and emit the LaTeX table + a
two-part narrative for the AMC paper.

The sweep measures torch.cuda.max_memory_allocated (MB) for AMC truncated
(M=8, O(M)) vs full BPTT (M=T, O(T)) on MUTAG at T in {32, 64}.

HONEST READING of the measured data (this is what the experiment actually
showed, and it is what we report):
  (1) MEMORY: the measured peak bytes are nearly identical between truncated
      (M=8) and full (M=T) at both T=32 (287 vs 272 MB) and T=64 (556 vs 523
      MB). Reason: the forward pass stacks all T hidden states for temporal
      pooling (spikergnn/stmp.py: torch.stack(hyp_accumulator) -> (T,N,d_h)),
      so live allocation is O(T) regardless of the truncation window M. The
      O(M) bound therefore governs the number of stored BACKWARD-HISTORY states
      (Table~ref{tab:scaling}), not the measured peak bytes in THIS
      implementation; real byte-level savings require forward chunking /
      gradient checkpointing (future work).
  (2) STABILITY: the truncated regime is numerically stable at T=64 (69.4% after
      3 epochs) whereas full BPTT aborts with a NaN at the same horizon -- the
      windowed metric correction + contraction decay (not the memory window
      itself) is what keeps long-horizon training stable, consistent with the
      bounded-error claim.

Each cell is a single seed (truncated: 3 epochs; full: 2 epochs, peak captured
before a NaN abort). Peak memory is reached within one backward pass and is
epoch-independent, so the comparison is valid.

Usage:
  C:/Python314/python.exe gen_measured_table.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "T_sweep_MUTAG.json")


def _fmean(xs):
    return sum(xs) / len(xs)


def main():
    d = json.load(open(PATH))
    res = d["results"]["MUTAG"]

    cells = {}      # (T, reg) -> mean peak MB
    crashed = {}    # (T, reg) -> bool (NaN abort before peak captured)
    accs = {}       # (T, reg) -> first acc (non-converged probe)
    for Tk in res:
        T = int(Tk[1:])
        for reg in res[Tk]:
            c = res[Tk][reg]
            peaks = [p for p in c.get("peak_mem_mb", []) if p is not None]
            if not peaks:
                continue
            cells[(T, reg)] = _fmean(peaks)
            has_crash = any(e.get("crashed_peak") for e in c.get("errors", []))
            crashed[(T, reg)] = has_crash and not c.get("accs")
            a = c.get("accs") or [None]
            accs[(T, reg)] = a[0]

    print("% measured peak MB (mean of per-run maxima):")
    for (T, reg), mb in sorted(cells.items()):
        print(f"%   T={T:3d} {reg:10s} peak={mb:8.1f} MB  "
              f"acc={accs.get((T,reg))}  crashed={crashed.get((T,reg))}")

    # table: T | Regime | Window M | Peak mem (MB) | Training outcome
    rows = []
    for T in (32, 64):
        tr = cells.get((T, "truncated"))
        fu = cells.get((T, "full"))
        if tr is None or fu is None:
            print(f"% WARNING: missing cell for T={T}")
            continue
        tr_out = f"{accs[(T,'truncated')]*100:.1f}\\% (3 ep)"
        fu_out = (f"NaN abort (pre-crash peak)"
                  if crashed.get((T, "full"))
                  else f"{accs[(T,'full')]*100:.1f}\\% (2 ep)")
        rows.append(
            f"{T} & AMC truncated & $M=8$ & {tr:7.1f} & stable, {tr_out} \\\\\n"
            f"{T} & Full BPTT (oracle) & $M={T}$ & {fu:7.1f} & {fu_out} \\\\"
        )

    table = (
        "\\begin{table}[htbp]\n"
        "\\caption{Measured peak GPU memory and training outcome of AMC "
        "truncated ($M{=}8$, $O(M)$) versus the full-BPTT oracle ($M{=}T$, "
        "$O(T)$) on MUTAG at longer horizons. Allocation peaks inside a single "
        "backward pass, so the value is independent of the epoch budget. "
        "Truncated cells: single seed, 3 epochs. Full-BPTT cells: single seed, "
        "2 epochs; the $T{=}64$ cell aborts with a NaN and its reported peak is "
        "the pre-crash maximum, whereas the $T{=}32$ cell completes. "
        "\\textbf{Top:} the measured peak bytes are nearly identical between the "
        "two regimes at both $T$ (e.g. 556 vs 523 MB at $T{=}64$): because the "
        "forward pass stacks all $T$ hidden states for temporal pooling "
        "(spikergnn/stmp.py), live allocation is $O(T)$ regardless of $M$, so "
        "the $O(M)$ bound governs stored backward-history states, not measured "
        "peak bytes in this implementation. \\textbf{Bottom:} the truncated "
        "regime stays numerically stable at $T{=}64$ while full BPTT aborts with "
        "a NaN -- the windowed metric correction and contraction decay, not the "
        "memory window, keep long-horizon training stable.}\n"
        "\\label{tab:measured_om}\n"
        "\\centering\\footnotesize\n"
        "\\begin{tabular}{@{}lcccc@{}}\n"
        "\\toprule\n"
        "$T$ & \\textbf{Regime} & \\textbf{Window $M$} & "
        "\\textbf{Peak mem (MB)} & \\textbf{Training outcome} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    print("\n=== LATEX TABLE ===\n")
    print(table)

    print(
        "\n=== NARRATIVE (two-part, honest) ===\n"
        "(1) MEMORY: measured peak bytes are nearly identical between truncated "
        "(M=8) and full BPTT (M=T) -- 287 vs 272 MB at T=32 and 556 vs 523 MB at "
        "T=64. The forward pass stacks all T hidden states for temporal pooling, "
        "so live allocation is O(T) independent of M; the O(M) bound governs the "
        "number of stored backward-history states (Table~\\ref{tab:scaling}), not "
        "the measured peak bytes here. Real byte-level savings need forward "
        "chunking / gradient checkpointing (future work).\n"
        "(2) STABILITY: truncated is numerically stable at T=64 (69.4% after 3 "
        "epochs) whereas full BPTT aborts with a NaN at the same horizon -- the "
        "windowed metric correction + contraction decay keeps long-horizon "
        "training stable, which is the mechanism behind the bounded-error claim."
    )


if __name__ == "__main__":
    main()
