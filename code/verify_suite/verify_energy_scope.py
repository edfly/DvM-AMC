"""
verify_energy_scope.py — P1-2 honesty check (machine-verifiable energy table).

Reproduces tab:energy_pareto (v55/v56) full-scope (MSPE+STMP) and STMP-only
energy reductions from SEPARATED spike counts returned by run_single_experiment
(res["mspe_spikes"], res["stmp_spikes"]), and compares against the hand-computed
table values.

The paper (Sec.6.3) explicitly states these are ANALYTICAL estimates (0.9 pJ SNN /
4.6 pJ ANN), not hardware measurements, and that the MSPE encoder fires at a
structural rate; this script makes the MSPE+STMP split machine-verifiable.

Protocol: 3-epoch probe (matches the P0-1 sweep), single seed. Absolute spike
counts are epoch-independent (per-pass), so the reduction % is a fair read of the
table's magnitude. Tolerance = 2.0 pp (run-to-run + probe-vs-100ep slack).

Exit 0 = reproduced within tolerance (safe to deliver).
"""
import os
import sys
import json

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.train import run_single_experiment
from spikergnn.energy import estimate_energy, count_model_params
from msg_baseline.config import DATASET_CONFIGS as _DSC

# ── NCI1 dataset-config injection (same fix as run_T_sweep.py) ──────────────
if "NCI1" not in _DSC:
    _DSC["NCI1"] = {
        "n_way": 2, "shots": [1, 5], "n_graphs": 4110,
        "n_classes": 2, "tu_name": "NCI1", "n_features": 37,
    }

# tab:energy_pareto (v55/v56) — full scope (MSPE+STMP) and STMP-only columns.
TABLE_FULL = {"MUTAG": 99.2, "PROTEINS": 98.5, "NCI1": 99.2, "DD": 89.4}
TABLE_STMP_ONLY = {"MUTAG": 99.6, "PROTEINS": 99.4, "NCI1": 99.9, "DD": 96.1}

TOL = 2.0  # percentage points
T = 8      # energy table uses M=T=8
DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD"]
EPOCHS = 3


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    cfg = SpikeRPGNNConfig(
        ablation="full", learnable_c=False, c_init=1.0,
        use_dvm=True, use_metric_correction=True,
        T=T, dvm_mode="truncated", dv_m_window=T, epochs=EPOCHS,
    )

    fail = 0
    out = {}
    for ds in DATASETS:
        in_dim = max(_DSC[ds].get("n_features", 7), 1)
        try:
            res = run_single_experiment(
                dataset_name=ds, shot=1, run_id=0, config=cfg, device=device,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {ds}: ERROR {repr(e)[:200]}", flush=True)
            fail = 1
            continue

        params = count_model_params(
            __import__("spikergnn.model", fromlist=["create_spikergnn"])
            .create_spikergnn(cfg, in_dim=7)
        )
        stmp = int(res.get("stmp_spikes", 0) or 0)
        mspe = int(res.get("mspe_spikes", 0) or 0)
        total = stmp + mspe
        rf = estimate_energy(total, params, T)["reduction_pct"]
        rs = estimate_energy(stmp, params, T)["reduction_pct"]
        tf = TABLE_FULL[ds]
        ts = TABLE_STMP_ONLY[ds]
        ok_f = abs(rf - tf) <= TOL
        ok_s = abs(rs - ts) <= TOL
        if not (ok_f and ok_s):
            fail = 1
        out[ds] = {
            "params": params, "stmp_spikes": stmp, "mspe_spikes": mspe,
            "total_spikes": total,
            "full_reduction_pct": round(rf, 2), "table_full": tf, "ok_full": ok_f,
            "stmp_only_reduction_pct": round(rs, 2), "table_stmp_only": ts, "ok_stmp": ok_s,
        }
        print(f"  {ds}: total={total} (stmp={stmp}, mspe={mspe}) | "
              f"full={rf:.2f}% (table {tf}, {ok_f}) | stmp-only={rs:.2f}% "
              f"(table {ts}, {ok_s})", flush=True)

    with open(os.path.join(_PROJECT_ROOT, "energy_scope_verify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("RESULT:", "ALL MATCH" if fail == 0 else "MISMATCH — fix table or code")
    sys.exit(fail)


if __name__ == "__main__":
    main()
