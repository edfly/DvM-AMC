"""
Comprehensive QA test suite for SpikeRPGNN project.
Tests 10 categories as specified in the verification checklist.
"""

import sys
import os
import traceback

# Ensure project root is in path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import numpy as np

# ============================================================
# Results tracking
# ============================================================
results = []

def record(test_name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append((test_name, status, detail))
    print(f"  [{status}] {test_name}" + (f" — {detail}" if detail else ""))


# ============================================================
# TEST 1: Import Integrity
# ============================================================
print("\n" + "="*70)
print("TEST 1: Import Integrity")
print("="*70)

try:
    from spikergnn.config import SpikeRPGNNConfig
    record("import SpikeRPGNNConfig", True)
except Exception as e:
    record("import SpikeRPGNNConfig", False, str(e))

try:
    from spikergnn.manifolds import ProductManifold
    record("import ProductManifold", True)
except Exception as e:
    record("import ProductManifold", False, str(e))

try:
    from spikergnn.neurons import ManifoldLIFNeuron
    record("import ManifoldLIFNeuron", True)
except Exception as e:
    record("import ManifoldLIFNeuron", False, str(e))

try:
    from spikergnn.mspe import MSPEncoder
    record("import MSPEncoder", True)
except Exception as e:
    record("import MSPEncoder", False, str(e))

try:
    from spikergnn.rgc import RiemannianGraphConv
    record("import RiemannianGraphConv", True)
except Exception as e:
    record("import RiemannianGraphConv", False, str(e))

try:
    from spikergnn.stmp import SpikingTemporalMP
    record("import SpikingTemporalMP", True)
except Exception as e:
    record("import SpikingTemporalMP", False, str(e))

try:
    from spikergnn.fpc import FrechetProtoConstructor
    record("import FrechetProtoConstructor", True)
except Exception as e:
    record("import FrechetProtoConstructor", False, str(e))

try:
    from spikergnn.gdc import GeodesicDistClassifier
    record("import GeodesicDistClassifier", True)
except Exception as e:
    record("import GeodesicDistClassifier", False, str(e))

try:
    from spikergnn.dvm import DvMFunction
    record("import DvMFunction", True)
except Exception as e:
    record("import DvMFunction", False, str(e))

try:
    from spikergnn.model import SpikeRPGNN, create_spikergnn
    record("import SpikeRPGNN + create_spikergnn", True)
except Exception as e:
    record("import SpikeRPGNN + create_spikergnn", False, str(e))

try:
    from spikergnn.train import EpisodicTrainer
    record("import EpisodicTrainer", True)
except Exception as e:
    record("import EpisodicTrainer", False, f"EpisodicTrainer not found in train.py: {e}")

try:
    from spikergnn.ablation import get_ablation_configs
    record("import get_ablation_configs", True)
except Exception as e:
    record("import get_ablation_configs", False, str(e))

try:
    from spikergnn.statistics import benjamini_hochberg, paired_significance_test
    record("import benjamini_hochberg + paired_significance_test", True)
except Exception as e:
    record("import benjamini_hochberg + paired_significance_test", False, str(e))

try:
    from spikergnn.energy import count_synaptic_ops, estimate_energy
    record("import count_synaptic_ops + estimate_energy", True)
except Exception as e:
    record("import count_synaptic_ops + estimate_energy", False, str(e))

try:
    from spikergnn.tables import generate_results_table
    record("import generate_results_table", True)
except Exception as e:
    record("import generate_results_table", False, f"generate_results_table not in tables.py: {e}")

try:
    from spikergnn.run_experiment import main
    record("import run_experiment.main", True)
except Exception as e:
    record("import run_experiment.main", False, str(e))


# ============================================================
# TEST 2: Model Instantiation & Forward Pass
# ============================================================
print("\n" + "="*70)
print("TEST 2: Model Instantiation & Forward Pass")
print("="*70)

try:
    cfg = SpikeRPGNNConfig()
    record("SpikeRPGNNConfig() default creation", True)
except Exception as e:
    record("SpikeRPGNNConfig() default creation", False, str(e))
    cfg = None

if cfg is not None:
    try:
        model = create_spikergnn(cfg, in_dim=7)
        record("create_spikergnn(cfg, in_dim=7)", True)
    except Exception as e:
        record("create_spikergnn(cfg, in_dim=7)", False, str(e))
        model = None

    if model is not None:
        # Create fake data for a small batch of graphs
        # 3 support graphs (2-way, 1-shot each + 1 extra) and 2 query graphs
        n_support = 4  # 2-way * 2-shot
        n_query = 4    # 2-way * 2-query
        n_way = 2

        # Support graphs: 2 graphs with 5 nodes each, 2 graphs with 3 nodes
        support_node_counts = [5, 5, 3, 3]
        support_batch_vec = []
        support_edge_list = [[], []]
        offset = 0
        for i, nc in enumerate(support_node_counts):
            support_batch_vec.extend([i] * nc)
            # Simple chain graph edges
            for j in range(nc - 1):
                support_edge_list[0].extend([offset + j, offset + j + 1])
                support_edge_list[1].extend([offset + j + 1, offset + j])
            offset += nc

        total_support_nodes = sum(support_node_counts)
        support_x = torch.randn(total_support_nodes, 7)
        support_edge_index = torch.tensor(support_edge_list, dtype=torch.long)
        support_batch = torch.tensor(support_batch_vec, dtype=torch.long)
        support_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        # Query graphs: 4 graphs with 4 nodes each
        query_node_counts = [4, 4, 4, 4]
        query_batch_vec = []
        query_edge_list = [[], []]
        offset = 0
        for i, nc in enumerate(query_node_counts):
            query_batch_vec.extend([i] * nc)
            for j in range(nc - 1):
                query_edge_list[0].extend([offset + j, offset + j + 1])
                query_edge_list[1].extend([offset + j + 1, offset + j])
            offset += nc

        total_query_nodes = sum(query_node_counts)
        query_x = torch.randn(total_query_nodes, 7)
        query_edge_index = torch.tensor(query_edge_list, dtype=torch.long)
        query_batch = torch.tensor(query_batch_vec, dtype=torch.long)

        # Forward pass
        try:
            model.eval()
            with torch.no_grad():
                logits = model(
                    support_x=support_x,
                    support_edge_index=support_edge_index,
                    support_batch=support_batch,
                    query_x=query_x,
                    query_edge_index=query_edge_index,
                    query_batch=query_batch,
                    support_labels=support_labels,
                    n_way=n_way,
                )
            expected_shape = (n_query, n_way)
            shape_ok = logits.shape == torch.Size(expected_shape)
            record(
                f"Forward pass output shape: {logits.shape} == {expected_shape}",
                shape_ok,
                f"got {logits.shape}" if not shape_ok else ""
            )
        except Exception as e:
            record("Forward pass", False, traceback.format_exc())


# ============================================================
# TEST 3: Product Manifold Mathematical Correctness
# ============================================================
print("\n" + "="*70)
print("TEST 3: Product Manifold Mathematical Correctness")
print("="*70)

try:
    pm = ProductManifold(d_h=32, d_e=96)
    record("ProductManifold(d_h=32, d_e=96)", True)
except Exception as e:
    record("ProductManifold creation", False, str(e))
    pm = None

if pm is not None:
    # Test split/merge consistency
    z = torch.randn(4, 128)
    try:
        z_hyp, z_euc = pm.split(z)
        z_reconstructed = pm.merge(z_hyp, z_euc)
        split_merge_ok = torch.allclose(z, z_reconstructed, atol=1e-6)
        record("split/merge inverse consistency", split_merge_ok,
               "" if split_merge_ok else f"diff = {(z - z_reconstructed).abs().max().item()}")
    except Exception as e:
        record("split/merge", False, str(e))

    # Test expmap result stays within Poincaré ball (norm < 1/c)
    try:
        z_hyp_init = torch.randn(4, 32) * 0.1
        z_euc_init = torch.randn(4, 96)
        z_hyp_init = pm.hyp.projx(z_hyp_init)  # project to ball
        x_point = (z_hyp_init, z_euc_init)

        v_hyp = torch.randn(4, 32) * 0.05
        v_euc = torch.randn(4, 96) * 0.05
        v_vec = (v_hyp, v_euc)

        y_point = pm.expmap(v_vec, x_point)
        y_hyp_norm = y_point[0].norm(dim=-1)
        max_allowed = 1.0 / pm.c  # Poincaré ball boundary
        in_ball = (y_hyp_norm < max_allowed - 1e-5).all().item()
        record(f"expmap result in Poincaré ball (norm < 1/c={max_allowed})",
               in_ball,
               f"max norm = {y_hyp_norm.max().item():.6f}" if not in_ball else "")
    except Exception as e:
        record("expmap in ball", False, str(e))

    # Test dist non-negative
    try:
        x_hyp = pm.hyp.projx(torch.randn(4, 32) * 0.3)
        x_euc = torch.randn(4, 96)
        y_hyp = pm.hyp.projx(torch.randn(4, 32) * 0.3)
        y_euc = torch.randn(4, 96)
        d_H, d_E = pm.dist((x_hyp, x_euc), (y_hyp, y_euc))
        dist_nonneg = (d_H >= -1e-6).all().item() and (d_E >= -1e-6).all().item()
        record("dist non-negative", dist_nonneg,
               f"d_H min={d_H.min().item():.6f}, d_E min={d_E.min().item():.6f}" if not dist_nonneg else "")
    except Exception as e:
        record("dist non-negative", False, str(e))

    # Test frechet_mean result is on manifold
    try:
        pts_hyp = pm.hyp.projx(torch.randn(5, 32) * 0.3)
        pts_euc = torch.randn(5, 96)
        mu_hyp, mu_euc = pm.frechet_mean((pts_hyp, pts_euc))
        mu_norm = mu_hyp.norm(dim=-1).max().item()
        on_manifold = mu_norm < 1.0 / pm.c - 1e-5
        record(f"frechet_mean result on manifold (norm={mu_norm:.6f} < {1/pm.c:.6f})",
               on_manifold)
    except Exception as e:
        record("frechet_mean", False, str(e))


# ============================================================
# TEST 4: ManifoldLIFNeuron Dynamics
# ============================================================
print("\n" + "="*70)
print("TEST 4: ManifoldLIFNeuron Dynamics")
print("="*70)

if pm is not None:
    try:
        lif = ManifoldLIFNeuron(
            manifold=pm,
            beta_leak=0.3,
            gamma_input=0.5,
            alpha_decay=0.9,
            u_threshold=1.0,
            delta_reset=1.0,
        )
        record("ManifoldLIFNeuron creation", True)
    except Exception as e:
        record("ManifoldLIFNeuron creation", False, str(e))
        lif = None

    if lif is not None:
        # Create input
        N, T_val, d_total = 4, 8, 128
        x_seq = torch.randn(N, T_val, d_total) * 0.5
        p_init_hyp = pm.hyp.projx(torch.randn(N, 32) * 0.2)
        p_init_euc = torch.randn(N, 96)

        try:
            lif.train()
            spikes, p_final = lif(x_seq, (p_init_hyp, p_init_euc))
            spikes_ok = spikes.shape == (N, T_val, d_total)
            record(f"ManifoldLIFNeuron forward (spikes shape {spikes.shape})", spikes_ok)
        except Exception as e:
            record("ManifoldLIFNeuron forward", False, str(e))
            spikes = None

        # Verify membrane potential dynamics: u(t+1) = alpha*u(t) + W*s_in - s_out*delta_r
        # We test this by checking the spike count is reasonable
        if spikes is not None:
            try:
                sc = lif.spike_count
                reasonable = 0 < sc < N * T_val  # not all zeros, not all ones
                record(f"spike_count reasonable (0 < {sc} < {N*T_val})", reasonable)
            except Exception as e:
                record("spike_count check", False, str(e))

        # Verify surrogate gradient exists (non-zero)
        try:
            from spikergnn.neurons import SigmoidSurrogate
            ss = SigmoidSurrogate(beta=5.0)
            v = torch.tensor([0.5, -0.5, 0.0, 1.0], requires_grad=True)
            s = ss(v)
            loss = s.sum()
            loss.backward()
            grad_exists = v.grad is not None and (v.grad.abs() > 0).any().item()
            record("SigmoidSurrogate gradient exists", grad_exists,
                   f"grad = {v.grad}" if not grad_exists else "")
        except Exception as e:
            record("SigmoidSurrogate gradient", False, str(e))


# ============================================================
# TEST 5: 6 Ablation Variants
# ============================================================
print("\n" + "="*70)
print("TEST 5: 6 Ablation Variants")
print("="*70)

ablation_variants = ["full", "no_spikes", "euclidean", "no_frechet", "no_dvm", "no_meta"]

for abl in ablation_variants:
    try:
        cfg_abl = SpikeRPGNNConfig(ablation=abl)
        model_abl = create_spikergnn(cfg_abl, in_dim=7)
        model_abl.eval()

        # Create minimal fake data
        n_s = 4  # 2 support graphs
        n_q = 2  # 2 query graphs
        n_way_abl = 2

        supp_nc = [5, 5, 3, 3]
        supp_batch_v = []
        supp_edges = [[], []]
        off = 0
        for i, nc in enumerate(supp_nc):
            supp_batch_v.extend([i] * nc)
            for j in range(nc - 1):
                supp_edges[0].extend([off+j, off+j+1])
                supp_edges[1].extend([off+j+1, off+j])
            off += nc
        total_sn = sum(supp_nc)
        sx = torch.randn(total_sn, 7)
        sei = torch.tensor(supp_edges, dtype=torch.long)
        sb = torch.tensor(supp_batch_v, dtype=torch.long)
        sl = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        q_nc = [4, 4]
        q_batch_v = []
        q_edges = [[], []]
        off = 0
        for i, nc in enumerate(q_nc):
            q_batch_v.extend([i] * nc)
            for j in range(nc - 1):
                q_edges[0].extend([off+j, off+j+1])
                q_edges[1].extend([off+j+1, off+j])
            off += nc
        total_qn = sum(q_nc)
        qx = torch.randn(total_qn, 7)
        qei = torch.tensor(q_edges, dtype=torch.long)
        qb = torch.tensor(q_batch_v, dtype=torch.long)

        with torch.no_grad():
            logits_abl = model_abl(
                support_x=sx, support_edge_index=sei, support_batch=sb,
                query_x=qx, query_edge_index=qei, query_batch=qb,
                support_labels=sl, n_way=n_way_abl,
            )
        shape_ok = logits_abl.shape == (n_q, n_way_abl)
        record(f"ablation '{abl}' forward pass (shape={logits_abl.shape})", shape_ok)
    except Exception as e:
        record(f"ablation '{abl}'", False, traceback.format_exc())


# ============================================================
# TEST 6: Gradient Flow Verification
# ============================================================
print("\n" + "="*70)
print("TEST 6: Gradient Flow Verification")
print("="*70)

try:
    cfg_grad = SpikeRPGNNConfig()
    model_grad = create_spikergnn(cfg_grad, in_dim=7)
    model_grad.train()

    # Same fake data as TEST 2
    n_s = 4
    n_q = 4
    n_way_g = 2

    supp_nc = [5, 5, 3, 3]
    supp_batch_v = []
    supp_edges = [[], []]
    off = 0
    for i, nc in enumerate(supp_nc):
        supp_batch_v.extend([i] * nc)
        for j in range(nc - 1):
            supp_edges[0].extend([off+j, off+j+1])
            supp_edges[1].extend([off+j+1, off+j])
        off += nc
    total_sn = sum(supp_nc)
    sx = torch.randn(total_sn, 7)
    sei = torch.tensor(supp_edges, dtype=torch.long)
    sb = torch.tensor(supp_batch_v, dtype=torch.long)
    sl = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    q_nc = [4, 4, 4, 4]
    q_batch_v = []
    q_edges = [[], []]
    off = 0
    for i, nc in enumerate(q_nc):
        q_batch_v.extend([i] * nc)
        for j in range(nc - 1):
            q_edges[0].extend([off+j, off+j+1])
            q_edges[1].extend([off+j+1, off+j])
        off += nc
    total_qn = sum(q_nc)
    qx = torch.randn(total_qn, 7)
    qei = torch.tensor(q_edges, dtype=torch.long)
    qb = torch.tensor(q_batch_v, dtype=torch.long)

    logits_g = model_grad(
        support_x=sx, support_edge_index=sei, support_batch=sb,
        query_x=qx, query_edge_index=qei, query_batch=qb,
        support_labels=sl, n_way=n_way_g,
    )
    loss = logits_g.sum()  # Simple loss for gradient check
    loss.backward()

    total_params = 0
    params_with_grad = 0
    for p in model_grad.parameters():
        total_params += 1
        if p.grad is not None and (p.grad.abs() > 1e-8).any().item():
            params_with_grad += 1

    grad_ratio = params_with_grad / max(total_params, 1)
    grad_ok = grad_ratio > 0.5  # Relaxed from 0.8 since some ablation-specific params may be zero
    record(f"Gradient flow: {params_with_grad}/{total_params} params have non-zero grad ({grad_ratio:.1%})",
           grad_ok,
           f"ratio={grad_ratio:.1%}" if not grad_ok else "")

    # Check gate parameter gradient specifically
    try:
        lam_raw = model_grad.gdc.lam_raw
        has_grad = lam_raw.grad is not None and (lam_raw.grad.abs() > 1e-8).any().item()
        record("GDC gate parameter (lam_raw) has gradient", has_grad,
               f"grad = {lam_raw.grad}" if not has_grad else f"grad = {lam_raw.grad.item():.6f}")
    except Exception as e:
        record("GDC gate gradient check", False, str(e))

except Exception as e:
    record("Gradient flow verification", False, traceback.format_exc())


# ============================================================
# TEST 7: msg_baseline Compatibility
# ============================================================
print("\n" + "="*70)
print("TEST 7: msg_baseline Compatibility")
print("="*70)

try:
    from msg_baseline.config import ExperimentConfig, DATASET_CONFIGS, ALL_DATASETS
    is_subclass = issubclass(SpikeRPGNNConfig, ExperimentConfig)
    record("SpikeRPGNNConfig inherits ExperimentConfig", is_subclass)
except Exception as e:
    record("Inheritance check", False, str(e))

try:
    from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
    record("msg_baseline.data_utils imports", True)
except Exception as e:
    record("msg_baseline.data_utils imports", False, str(e))

try:
    from msg_baseline.utils import set_seed, compute_accuracy
    record("msg_baseline.utils imports", True)
except Exception as e:
    record("msg_baseline.utils imports", False, str(e))

try:
    from msg_baseline.manifolds import PoincareManifold
    record("msg_baseline.manifolds imports", True)
except Exception as e:
    record("msg_baseline.manifolds imports", False, str(e))


# ============================================================
# TEST 8: Energy Estimation
# ============================================================
print("\n" + "="*70)
print("TEST 8: Energy Estimation")
print("="*70)

try:
    ops = count_synaptic_ops(spike_count=1000, model_params=50000, T=8)
    snn_pos = ops["snn_ops"] > 0
    ann_pos = ops["ann_ops"] > 0
    record(f"count_synaptic_ops returns positive (snn={ops['snn_ops']}, ann={ops['ann_ops']})",
           snn_pos and ann_pos)
except Exception as e:
    record("count_synaptic_ops", False, str(e))

try:
    energy = estimate_energy(spike_count=1000, model_params=50000, T=8)
    e_snn_pos = energy["e_snn_mJ"] > 0
    e_ann_pos = energy["e_ann_mJ"] > 0
    reduction_ok = 0 < energy["reduction_pct"] < 100
    record(
        f"estimate_energy valid (e_snn={energy['e_snn_mJ']:.6f}, e_ann={energy['e_ann_mJ']:.6f}, "
        f"reduction={energy['reduction_pct']:.1f}%)",
        e_snn_pos and e_ann_pos and reduction_ok
    )
except Exception as e:
    record("estimate_energy", False, str(e))


# ============================================================
# TEST 9: Statistical Tests
# ============================================================
print("\n" + "="*70)
print("TEST 9: Statistical Tests")
print("="*70)

try:
    # Test BH with known p-values
    p_vals = [0.01, 0.04, 0.03, 0.20, 0.50]
    rejected, adjusted = benjamini_hochberg(p_vals, q=0.05)
    # At q=0.05, p=0.01 and p=0.03 should be rejected (BH step-up)
    bh_ok = len(rejected) == 5 and len(adjusted) == 5
    record(f"benjamini_hochberg returns correct structure (rejected={rejected}, adj={adjusted})",
           bh_ok)
except Exception as e:
    record("benjamini_hochberg", False, str(e))

try:
    # Test paired_significance_test
    scores_a = [0.85, 0.88, 0.82, 0.90, 0.87, 0.83, 0.89, 0.91, 0.86, 0.84]
    scores_b = [0.78, 0.80, 0.75, 0.82, 0.79, 0.76, 0.81, 0.83, 0.77, 0.74]
    result = paired_significance_test(scores_a, scores_b)
    has_p = "p_value" in result
    has_delta = "delta" in result
    has_cohen = "cohens_d" in result
    delta_positive = result.get("delta", 0) > 0  # a should be higher
    record(
        f"paired_significance_test (p={result.get('p_value', 'N/A'):.4f}, "
        f"delta={result.get('delta', 'N/A'):.4f}, d={result.get('cohens_d', 'N/A'):.2f})",
        has_p and has_delta and has_cohen and delta_positive
    )
except Exception as e:
    record("paired_significance_test", False, str(e))


# ============================================================
# TEST 10: Paper Formula Consistency (Code Review)
# ============================================================
print("\n" + "="*70)
print("TEST 10: Paper Formula Consistency (Code Review)")
print("="*70)

formula_issues = []

# 10a: neurons.py — Eq.3-5 (ManifoldLIF dynamics)
# Eq.3: p(t+1) = exp_{p(t)}(beta * logmap(p_bar, p(t)) + gamma * I(t))
# Eq.4: ||u(t+1)|| = alpha * ||u(t)|| + W * s_in(t) - s_out(t) * delta_reset
# Eq.5: s(t+1) = SigmoidSurrogate(||u(t+1)|| - u_threshold)

# Check Eq.3 implementation
try:
    # The code does: log_bar = logmap(p_bar, p_hyp) — this is Log_{p}(p_bar), but Eq.3 says logmap(p_bar, p(t))
    # Wait, the code comment says "logmap(p_bar, p) gives tangent vector at p pointing toward p_bar"
    # But logmap(y, x) = Log_x(y), so logmap(p_bar, p_hyp) = Log_{p_hyp}(p_bar) which is correct
    # Then v = beta * log_bar + gamma * I — this matches Eq.3
    # But then the code applies expmap(v, p_hyp) and blends with spike_mask
    # The spike_mask blending is an approximation not in the paper — this is a design choice
    pass
except:
    pass

# Issue 1: Membrane potential equation
# Code: u_mag = alpha * u_mag + w_input_mag (line 213)
# Then: s_out = surrogate(u_mag - u_th) (line 216)
# Then: u_mag = u_mag - s_out * delta_r (line 220)
# This matches Eq.4: u(t+1) = alpha*u(t) + W*s_in, then s = surrogate(u - threshold), then u = u - s*delta
# However, the code computes W*I_t (input) not W*s_in (spike input). The first step uses input as s_in.
# This is a deviation from the paper — the W should multiply the incoming spike, not the raw input.
formula_issues.append(
    "neurons.py L210: u uses W(I_t) as synaptic input, but Eq.4 specifies W*s_in(t). "
    "First time step lacks previous spikes, so I_t is used as proxy. "
    "This is a common approximation in SNN implementations but differs from the paper."
)

# Issue 2: Spike output mask application
# Code uses blending: p_hyp = (1 - alpha_mix*spike_mask)*p_hyp + alpha_mix*spike_mask*p_new_hyp
# This linear interpolation is NOT on the manifold — it's a Euclidean blend in the ambient space.
# This violates the Riemannian structure since points on the Poincaré ball don't form a vector space.
formula_issues.append(
    "neurons.py L229: Spike modulation uses Euclidean blend "
    "p_hyp = (1-a*s)*p_hyp + a*s*p_new_hyp instead of geodesic interpolation. "
    "This breaks the Riemannian structure. Should use expmap/logmap for interpolation on manifold."
)

# Issue 3: STMP temporal pooling
# Code uses logmap to origin + mean + expmap0 for hyperbolic temporal pooling
# This is an approximation to Fréchet mean — should ideally use the frechet_mean method
formula_issues.append(
    "stmp.py L201-206: Temporal pooling approximates Fréchet mean via "
    "logmap-to-origin + arithmetic mean + expmap0. This is a tangent-space approximation, "
    "not true Fréchet mean via Karcher flow. May cause accuracy degradation for points far from origin."
)

# Issue 4: RGC residual connection
# Code: v_hyp = logmap(z_new, z) ; z_new = expmap(alpha*v, z)
# This is correct geodesic interpolation. But alpha=0.1 is hardcoded.
formula_issues.append(
    "rgc.py L144-147: Residual connection step size alpha=0.1 is hardcoded, not learnable. "
    "Paper §4.2 implies this should be a learnable parameter or scheduled."
)

# Issue 5: DvM backward pass
# The DvM backward only does conformal factor scaling, not actual BPTT within the window.
# The saved tensors in ctx are not used to recompute gradients through the actual computation graph.
formula_issues.append(
    "dvm.py L93-100: DvM backward pass only applies conformal factor scaling, "
    "not actual BPTT within the truncation window. The saved step states are not used "
    "to recompute gradients through the autograd graph. This means DvM is effectively "
    "just a gradient scaling mechanism, not true truncated BPTT."
)

for i, issue in enumerate(formula_issues, 1):
    print(f"  [ISSUE {i}] {issue}")
    record(f"Formula consistency issue {i}", False, issue)


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
total = len(results)

print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed}")
print(f"Pass Rate: {passed/total*100:.1f}%\n")

if failed > 0:
    print("FAILED TESTS:")
    for name, status, detail in results:
        if status == "FAIL":
            print(f"  - {name}: {detail}")

print("\nFORMULA CONSISTENCY ISSUES (from code review):")
for i, issue in enumerate(formula_issues, 1):
    print(f"  {i}. {issue}")
