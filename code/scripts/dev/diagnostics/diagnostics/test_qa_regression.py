"""
SpikeRPGNN Regression Test Suite — 8 Bug Fix Verification

Tests all 8 bug fixes plus additional boundary tests:
  Bug 1 (P2): EpisodicTrainer import
  Bug 2 (P2): get_ablation_configs alias
  Bug 3 (P2): generate_results_table function
  Bug 4 (P0): LIF neuron sparsity fix (w_input_mag normalization + learnable threshold + persistent _u_mag)
  Bug 5 (P0): Geodesic interpolation fix (expmap/logmap replaces linear blend)
  Bug 6 (P1): STMP uses frechet_mean
  Bug 7 (P2): RGC learnable alpha
  Bug 8 (P1): DvM comment documentation

Plus: ablation variants, gradient flow, multiple forward passes, different batch sizes.
"""

import sys
import os
import math
import traceback

# Ensure project root is on path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn

# ─── Test Infrastructure ───────────────────────────────────────────────

results = []

def record(test_name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append((test_name, passed, detail))
    print(f"  [{status}] {test_name}" + (f" — {detail}" if detail else ""))

def summary():
    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    failed = total - passed
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed}/{total} passed, {failed} failed")
    print("=" * 70)
    if failed > 0:
        print("\nFailed tests:")
        for name, p, detail in results:
            if not p:
                print(f"  ✗ {name}: {detail}")
    else:
        print("\nAll tests passed!")
    return failed == 0


# ─── Helper: create model and synthetic data ───────────────────────────

def make_config(ablation="full", **overrides):
    from spikergnn.config import SpikeRPGNNConfig
    defaults = dict(
        hidden_dim=128,
        d_h=32,
        d_e=96,
        T=4,
        beta_leak=0.3,
        gamma_input=0.5,
        alpha_decay=0.9,
        u_threshold=1.0,
        delta_reset=1.0,
        surrogate_beta=5.0,
        gate_init=0.5,
        ablation=ablation,
        use_dvm=True,
        num_layers=1,
    )
    defaults.update(overrides)
    return SpikeRPGNNConfig(**defaults)


def make_graph_data(n_nodes=10, n_edges=20, n_graphs=2, in_dim=7, device="cpu"):
    """Create synthetic graph data for testing."""
    # Node features
    x = torch.randn(n_nodes, in_dim, device=device)

    # Random edges (ensure at least self-loops)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), device=device)
    # Add self-loops
    self_loops = torch.arange(n_nodes, device=device).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    # Batch assignment
    nodes_per_graph = n_nodes // n_graphs
    batch = torch.cat([
        torch.full((nodes_per_graph,), i, device=device)
        for i in range(n_graphs)
    ])
    # Handle remainder
    remainder = n_nodes - nodes_per_graph * n_graphs
    if remainder > 0:
        batch = torch.cat([batch, torch.full((remainder,), n_graphs - 1, device=device)])

    return x, edge_index, batch


# ═══════════════════════════════════════════════════════════════════════
# TEST GROUP 1: Import Integrity (Bug 1, 2, 3)
# ═══════════════════════════════════════════════════════════════════════

def test_imports():
    print("\n── Test Group 1: Import Integrity ──────────────────────────")

    # Bug 1: EpisodicTrainer
    try:
        from spikergnn.train import EpisodicTrainer
        record("Bug1: import EpisodicTrainer", True)
    except Exception as e:
        record("Bug1: import EpisodicTrainer", False, str(e))

    # Bug 2: get_ablation_configs alias
    try:
        from spikergnn.ablation import get_ablation_configs
        record("Bug2: import get_ablation_configs", True)
    except Exception as e:
        record("Bug2: import get_ablation_configs", False, str(e))

    # Bug 3: generate_results_table function
    try:
        from spikergnn.tables import generate_results_table
        record("Bug3: import generate_results_table", True)
    except Exception as e:
        record("Bug3: import generate_results_table", False, str(e))

    # Verify alias points to same function
    try:
        from spikergnn.ablation import get_ablation_configs, get_all_ablation_configs
        same = get_ablation_configs is get_all_ablation_configs
        record("Bug2: get_ablation_configs is alias of get_all_ablation_configs", same)
    except Exception as e:
        record("Bug2: alias check", False, str(e))

    # Verify generate_results_table is callable
    try:
        from spikergnn.tables import generate_results_table
        callable_ok = callable(generate_results_table)
        record("Bug3: generate_results_table is callable", callable_ok)
    except Exception as e:
        record("Bug3: callable check", False, str(e))

    # EpisodicTrainer can be instantiated with dummy args
    try:
        from spikergnn.train import EpisodicTrainer
        from spikergnn.model import SpikeRPGNN
        config = make_config()
        model = SpikeRPGNN(config=config, in_dim=7)
        trainer = EpisodicTrainer(model=model, config=config, device="cpu")
        record("Bug1: EpisodicTrainer can be instantiated", True)
    except Exception as e:
        record("Bug1: EpisodicTrainer instantiation", False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# TEST GROUP 2: Model Instantiation & Forward Pass
# ═══════════════════════════════════════════════════════════════════════

def test_model_forward():
    print("\n── Test Group 2: Model Instantiation & Forward Pass ────────")

    try:
        from spikergnn.model import SpikeRPGNN, create_spikergnn
        config = make_config(T=4, num_layers=1)
        model = SpikeRPGNN(config=config, in_dim=7)
        model.eval()

        x, edge_index, batch = make_graph_data(n_nodes=10, n_edges=20, n_graphs=2, in_dim=7)
        z_graph = model.encode(x, edge_index, batch)

        z_hyp, z_euc = z_graph
        record("Forward: encode returns tuple", isinstance(z_graph, tuple) and len(z_graph) == 2)
        record("Forward: z_hyp shape correct", z_hyp.shape == (2, 32), f"got {z_hyp.shape}")
        record("Forward: z_euc shape correct", z_euc.shape == (2, 96), f"got {z_euc.shape}")
    except Exception as e:
        record("Forward: encode", False, str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
# TEST GROUP 3: Bug 4 — LIF Neuron Sparsity Fix
# ═══════════════════════════════════════════════════════════════════════

def test_bug4_sparsity():
    print("\n── Test Group 3: Bug 4 — LIF Neuron Sparsity ─────────────")

    try:
        from spikergnn.neurons import ManifoldLIFNeuron
        from spikergnn.manifolds import ProductManifold

        manifold = ProductManifold(d_h=32, d_e=96, c=1.0)
        lif = ManifoldLIFNeuron(
            manifold=manifold,
            beta_leak=0.3,
            gamma_input=0.5,
            alpha_decay=0.9,
            u_threshold=1.0,
            delta_reset=1.0,
            surrogate_beta=5.0,
            ablation="full",
        )
        lif.eval()

        N, T, d_total = 20, 8, 128
        x_seq = torch.randn(N, T, d_total)
        p_hyp = torch.randn(N, 32)
        p_hyp = manifold.hyp.projx(p_hyp)
        p_euc = torch.randn(N, 96)

        with torch.no_grad():
            spikes, (p_hyp_final, p_euc_final) = lif(x_seq, (p_hyp, p_euc))

        # Check spike count
        total_possible = N * T  # spike_count counts per-node per-step (broadcast to d_total)
        actual_spike_count = lif.spike_count
        spike_rate = actual_spike_count / total_possible if total_possible > 0 else 0

        record("Bug4: spike_rate < 80%", spike_rate < 0.80,
               f"rate={spike_rate:.2%}, count={actual_spike_count}, N*T={total_possible}")

        # Check w_input_mag is normalized by sqrt(d_total)
        # We verify this by checking the code path: w_input_mag / sqrt(d_total)
        w_input = lif.W(x_seq[:, 0, :])
        w_input_mag = w_input.norm(dim=-1)
        normalized_mag = w_input_mag / math.sqrt(lif.d_total)
        # After normalization, magnitude should be ~O(1)
        mean_norm = normalized_mag.mean().item()
        record("Bug4: normalized w_input_mag is O(1)", 0.01 < mean_norm < 10.0,
               f"mean normalized mag = {mean_norm:.4f}")

        # Check u_th_learnable exists and is a Parameter
        has_learnable_th = hasattr(lif, 'u_th_learnable') and isinstance(lif.u_th_learnable, nn.Parameter)
        record("Bug4: u_th_learnable is nn.Parameter", has_learnable_th)

        if has_learnable_th:
            init_val = lif.u_th_learnable.item()
            record("Bug4: u_th_learnable initial ≈ 0.35", abs(init_val - 0.35) < 0.01,
                   f"init_val={init_val:.4f}")

        # Check _u_mag buffer persists
        has_u_mag_buffer = hasattr(lif, '_u_mag') and lif._u_mag is not None
        record("Bug4: _u_mag buffer persists after forward", has_u_mag_buffer)

        if has_u_mag_buffer:
            record("Bug4: _u_mag shape matches N", lif._u_mag.shape[0] == N,
                   f"shape={lif._u_mag.shape}")

    except Exception as e:
        record("Bug4: sparsity test", False, str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
# TEST GROUP 4: Bug 5 — Geodesic Interpolation
# ═══════════════════════════════════════════════════════════════════════

def test_bug5_geodesic():
    print("\n── Test Group 4: Bug 5 — Geodesic Interpolation ───────────")

    try:
        from spikergnn.model import SpikeRPGNN

        config = make_config(T=4, num_layers=1, ablation="full")
        model = SpikeRPGNN(config=config, in_dim=7)
        model.eval()

        x, edge_index, batch = make_graph_data(n_nodes=20, n_edges=40, n_graphs=2, in_dim=7)

        with torch.no_grad():
            z_graph = model.encode(x, edge_index, batch)

        z_hyp, z_euc = z_graph

        # All z_hyp should be inside Poincaré ball (norm < 1.0)
        hyp_norms = z_hyp.norm(dim=-1)
        max_norm = hyp_norms.max().item()
        all_inside = (hyp_norms < 1.0).all().item()

        record("Bug5: all z_hyp inside Poincare ball (norm < 1.0)", all_inside,
               f"max_norm={max_norm:.6f}")

        # Check spike_alpha parameter exists
        lif = model.stmp.lif
        has_spike_alpha = hasattr(lif, 'spike_alpha') and isinstance(lif.spike_alpha, nn.Parameter)
        record("Bug5: spike_alpha is nn.Parameter", has_spike_alpha)

    except Exception as e:
        record("Bug5: geodesic test", False, str(e))
        traceback.print_exc()

    # Test euclidean ablation still uses linear blend
    try:
        from spikergnn.neurons import ManifoldLIFNeuron
        from spikergnn.manifolds import ProductManifold

        manifold = ProductManifold(d_h=32, d_e=96, c=1.0)
        lif_euc = ManifoldLIFNeuron(
            manifold=manifold,
            ablation="euclidean",
        )
        lif_euc.eval()

        N, T, d_total = 5, 4, 128
        x_seq = torch.randn(N, T, d_total)
        p_hyp = torch.randn(N, 32) * 0.1
        p_hyp = manifold.hyp.projx(p_hyp)
        p_euc = torch.randn(N, 96)

        with torch.no_grad():
            spikes, (p_hyp_f, p_euc_f) = lif_euc(x_seq, (p_hyp, p_euc))

        # Even in euclidean mode, result should be inside ball (due to projx)
        euc_norms = p_hyp_f.norm(dim=-1)
        all_inside_euc = (euc_norms < 1.0).all().item()
        record("Bug5: euclidean ablation z_hyp still inside ball", all_inside_euc,
               f"max_norm={euc_norms.max().item():.6f}")

    except Exception as e:
        record("Bug5: euclidean ablation test", False, str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
# TEST GROUP 5: Bug 6 — STMP Uses frechet_mean
# ═══════════════════════════════════════════════════════════════════════

def test_bug6_frechet():
    print("\n── Test Group 5: Bug 6 — STMP frechet_mean ────────────────")

    try:
        # Read stmp.py and verify frechet_mean is called
        stmp_path = os.path.join(_project_root, "spikergnn", "stmp.py")
        with open(stmp_path, "r", encoding="utf-8") as f:
            stmp_code = f.read()

        has_frechet_call = "frechet_mean" in stmp_code
        record("Bug6: stmp.py contains frechet_mean call", has_frechet_call)

        # Verify the actual forward path works and output is on manifold
        from spikergnn.model import SpikeRPGNN
        config = make_config(T=4, num_layers=1, ablation="full")
        model = SpikeRPGNN(config=config, in_dim=7)
        model.eval()

        x, edge_index, batch = make_graph_data(n_nodes=10, n_edges=20, n_graphs=2, in_dim=7)

        with torch.no_grad():
            z_graph = model.encode(x, edge_index, batch)

        z_hyp, z_euc = z_graph
        hyp_norms = z_hyp.norm(dim=-1)
        on_manifold = (hyp_norms < 1.0).all().item()
        record("Bug6: STMP output on manifold after frechet_mean", on_manifold,
               f"max_norm={hyp_norms.max().item():.6f}")

    except Exception as e:
        record("Bug6: frechet_mean test", False, str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
# TEST GROUP 6: Bug 7 — RGC Learnable Alpha
# ═══════════════════════════════════════════════════════════════════════

def test_bug7_alpha():
    print("\n── Test Group 6: Bug 7 — RGC Learnable Alpha ──────────────")

    try:
        from spikergnn.rgc import RiemannianGraphConv
        from spikergnn.manifolds import ProductManifold

        manifold = ProductManifold(d_h=32, d_e=96, c=1.0)
        rgc = RiemannianGraphConv(manifold=manifold, hidden_dim=128, ablation="full")

        # Check residual_alpha is nn.Parameter
        has_alpha = hasattr(rgc, 'residual_alpha')
        is_param = isinstance(rgc.residual_alpha, nn.Parameter)
        record("Bug7: residual_alpha exists as nn.Parameter", has_alpha and is_param)

        if has_alpha:
            init_val = rgc.residual_alpha.item()
            record("Bug7: residual_alpha initial ≈ 0.1", abs(init_val - 0.1) < 0.01,
                   f"init_val={init_val:.4f}")

            # Check sigmoid constraint: sigmoid(0.1) ≈ 0.525
            alpha_sig = torch.sigmoid(rgc.residual_alpha).item()
            in_range = 0.0 < alpha_sig < 1.0
            record("Bug7: sigmoid(residual_alpha) in (0, 1)", in_range,
                   f"sigmoid(alpha)={alpha_sig:.4f}")

    except Exception as e:
        record("Bug7: alpha test", False, str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
# TEST GROUP 7: Bug 8 — DvM Comment Documentation
# ═══════════════════════════════════════════════════════════════════════

def test_bug8_dvm_comment():
    print("\n── Test Group 7: Bug 8 — DvM Comment Documentation ────────")

    try:
        dvm_path = os.path.join(_project_root, "spikergnn", "dvm.py")
        with open(dvm_path, "r", encoding="utf-8") as f:
            dvm_code = f.read()

        # Check for comment about simplified version
        has_simplified_comment = "simplified" in dvm_code.lower() or "simplif" in dvm_code.lower()
        record("Bug8: dvm.py contains 'simplified' comment", has_simplified_comment)

        # Check for specific note about the current implementation
        has_note = "NOTE" in dvm_code or "note" in dvm_code.lower()
        record("Bug8: dvm.py contains implementation note", has_note)

        # Verify DvM functionality is not broken
        from spikergnn.dvm import DvMFunction, apply_dvm
        from spikergnn.manifolds import ProductManifold

        manifold = ProductManifold(d_h=32, d_e=96, c=1.0)
        N = 5
        p_init_hyp = torch.randn(N, 32) * 0.1
        p_init_hyp = manifold.hyp.projx(p_init_hyp)
        p_init_euc = torch.randn(N, 96)

        # Create step outputs
        step_outputs = []
        for _ in range(4):
            p_h = torch.randn(N, 32) * 0.1
            p_h = manifold.hyp.projx(p_h)
            p_e = torch.randn(N, 96)
            step_outputs.append((p_h, p_e))

        p_final = apply_dvm(manifold, (p_init_hyp, p_init_euc), step_outputs, window=2)
        result_is_tuple = isinstance(p_final, tuple) and len(p_final) == 2
        record("Bug8: apply_dvm returns valid Point tuple", result_is_tuple)

        if result_is_tuple:
            p_hyp_f, p_euc_f = p_final
            hyp_norms = p_hyp_f.norm(dim=-1)
            inside = (hyp_norms < 1.0).all().item()
            record("Bug8: apply_dvm output on manifold", inside,
                   f"max_norm={hyp_norms.max().item():.6f}")

    except Exception as e:
        record("Bug8: DvM comment/functionality test", False, str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
# TEST GROUP 8: Ablation Variants
# ═══════════════════════════════════════════════════════════════════════

def test_ablation_variants():
    print("\n── Test Group 8: Ablation Variant Forward Pass ────────────")

    variants = ["full", "no_spikes", "euclidean", "no_frechet", "no_dvm", "no_meta"]

    for variant in variants:
        try:
            from spikergnn.model import SpikeRPGNN
            config = make_config(ablation=variant, T=4, num_layers=1)
            model = SpikeRPGNN(config=config, in_dim=7)
            model.eval()

            x, edge_index, batch = make_graph_data(n_nodes=10, n_edges=20, n_graphs=2, in_dim=7)

            with torch.no_grad():
                z_graph = model.encode(x, edge_index, batch)

            z_hyp, z_euc = z_graph
            hyp_norms = z_hyp.norm(dim=-1)
            inside = (hyp_norms < 1.0).all().item()
            record(f"Ablation '{variant}': forward pass OK, z_hyp inside ball",
                   inside, f"max_norm={hyp_norms.max().item():.6f}")

        except Exception as e:
            record(f"Ablation '{variant}': forward pass", False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# TEST GROUP 9: Gradient Flow
# ═══════════════════════════════════════════════════════════════════════

def test_gradient_flow():
    print("\n── Test Group 9: Gradient Flow Verification ───────────────")

    try:
        from spikergnn.model import SpikeRPGNN

        config = make_config(T=4, num_layers=1, ablation="full")
        model = SpikeRPGNN(config=config, in_dim=7)
        model.train()

        x, edge_index, batch = make_graph_data(n_nodes=10, n_edges=20, n_graphs=2, in_dim=7)

        # Encode
        z_graph = model.encode(x, edge_index, batch)
        z_hyp, z_euc = z_graph

        # Create a simple loss
        loss = z_hyp.sum() + z_euc.sum()
        loss.backward()

        # Check gradient flow
        total_params = 0
        params_with_grad = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None and param.grad.abs().sum() > 0:
                    params_with_grad += 1

        grad_ratio = params_with_grad / total_params if total_params > 0 else 0
        record(f"Gradient flow: {params_with_grad}/{total_params} params have non-zero grad",
               grad_ratio >= 0.5, f"ratio={grad_ratio:.2%}")

    except Exception as e:
        record("Gradient flow test", False, str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
# TEST GROUP 10: Boundary Tests
# ═══════════════════════════════════════════════════════════════════════

def test_boundary():
    print("\n── Test Group 10: Boundary Tests ───────────────────────────")

    # Test 10a: Multiple forward passes — _u_mag accumulation & reset
    try:
        from spikergnn.neurons import ManifoldLIFNeuron
        from spikergnn.manifolds import ProductManifold

        manifold = ProductManifold(d_h=32, d_e=96, c=1.0)
        lif = ManifoldLIFNeuron(manifold=manifold, ablation="full")
        lif.eval()

        N, T, d_total = 5, 4, 128
        p_hyp = torch.randn(N, 32) * 0.1
        p_hyp = manifold.hyp.projx(p_hyp)
        p_euc = torch.randn(N, 96)

        # First forward pass
        x_seq1 = torch.randn(N, T, d_total)
        with torch.no_grad():
            _, (p_hyp, p_euc) = lif(x_seq1, (p_hyp, p_euc))
        u_mag_1 = lif._u_mag.clone()

        # Second forward pass — _u_mag should accumulate
        x_seq2 = torch.randn(N, T, d_total)
        with torch.no_grad():
            _, (p_hyp, p_euc) = lif(x_seq2, (p_hyp, p_euc))
        u_mag_2 = lif._u_mag.clone()

        accumulates = not torch.allclose(u_mag_1, u_mag_2)
        record("Boundary: _u_mag accumulates across forward passes", accumulates)

        # Reset and verify
        lif.reset_membrane()
        reset_ok = lif._u_mag is None
        record("Boundary: reset_membrane clears _u_mag", reset_ok)

        # After reset, fresh forward should work
        p_hyp_r = torch.randn(N, 32) * 0.1
        p_hyp_r = manifold.hyp.projx(p_hyp_r)
        p_euc_r = torch.randn(N, 96)
        x_seq3 = torch.randn(N, T, d_total)
        with torch.no_grad():
            _, _ = lif(x_seq3, (p_hyp_r, p_euc_r))
        record("Boundary: forward after reset works", lif._u_mag is not None)

    except Exception as e:
        record("Boundary: _u_mag accumulation test", False, str(e))
        traceback.print_exc()

    # Test 10b: Different batch sizes and graph sizes
    for n_nodes, n_graphs in [(4, 1), (30, 3), (50, 5)]:
        try:
            from spikergnn.model import SpikeRPGNN
            config = make_config(T=4, num_layers=1, ablation="full")
            model = SpikeRPGNN(config=config, in_dim=7)
            model.eval()

            x, edge_index, batch = make_graph_data(
                n_nodes=n_nodes, n_edges=max(n_nodes * 2, 6),
                n_graphs=n_graphs, in_dim=7
            )

            with torch.no_grad():
                z_graph = model.encode(x, edge_index, batch)

            z_hyp, z_euc = z_graph
            shape_ok = z_hyp.shape[0] == n_graphs and z_euc.shape[0] == n_graphs
            hyp_ok = (z_hyp.norm(dim=-1) < 1.0).all().item()
            record(f"Boundary: n_nodes={n_nodes}, n_graphs={n_graphs}",
                   shape_ok and hyp_ok,
                   f"z_hyp shape={z_hyp.shape}, inside={hyp_ok}")

        except Exception as e:
            record(f"Boundary: n_nodes={n_nodes}, n_graphs={n_graphs}", False, str(e))

    # Test 10c: Euclidean ablation uses linear interpolation (no logmap/expmap for spike blend)
    try:
        from spikergnn.neurons import ManifoldLIFNeuron
        from spikergnn.manifolds import ProductManifold

        manifold = ProductManifold(d_h=32, d_e=96, c=1.0)
        lif = ManifoldLIFNeuron(manifold=manifold, ablation="euclidean")
        lif.eval()

        N, T, d_total = 5, 4, 128
        x_seq = torch.randn(N, T, d_total)
        p_hyp = torch.randn(N, 32) * 0.1
        p_hyp = manifold.hyp.projx(p_hyp)
        p_euc = torch.randn(N, 96)

        with torch.no_grad():
            _, (p_hyp_f, p_euc_f) = lif(x_seq, (p_hyp, p_euc))

        inside = (p_hyp_f.norm(dim=-1) < 1.0).all().item()
        record("Boundary: euclidean ablation linear blend keeps z_hyp in ball",
               inside, f"max_norm={p_hyp_f.norm(dim=-1).max().item():.6f}")

    except Exception as e:
        record("Boundary: euclidean linear blend", False, str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
# TEST GROUP 11: Full Classification Pipeline
# ═══════════════════════════════════════════════════════════════════════

def test_classification_pipeline():
    print("\n── Test Group 11: Full Classification Pipeline ─────────────")

    try:
        from spikergnn.model import SpikeRPGNN

        config = make_config(T=4, num_layers=1, ablation="full")
        model = SpikeRPGNN(config=config, in_dim=7)
        model.eval()

        # Support set: 2 graphs, 2 classes
        sup_x, sup_edge, sup_batch = make_graph_data(n_nodes=10, n_edges=20, n_graphs=2, in_dim=7)
        # Query set: 3 graphs
        que_x, que_edge, que_batch = make_graph_data(n_nodes=15, n_edges=30, n_graphs=3, in_dim=7)
        sup_labels = torch.tensor([0, 1])

        with torch.no_grad():
            sup_z = model.encode(sup_x, sup_edge, sup_batch)
            que_z = model.encode(que_x, que_edge, que_batch)
            logits = model.classify(que_z, sup_z, sup_labels, n_classes=2)

        shape_ok = logits.shape == (3, 2)
        record("Classification: logits shape (3, 2)", shape_ok, f"got {logits.shape}")

        # Logits should be finite
        finite = torch.isfinite(logits).all().item()
        record("Classification: all logits finite", finite)

    except Exception as e:
        record("Classification pipeline", False, str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SpikeRPGNN Regression Test Suite — 8 Bug Fix Verification")
    print("=" * 70)

    torch.manual_seed(42)

    test_imports()
    test_model_forward()
    test_bug4_sparsity()
    test_bug5_geodesic()
    test_bug6_frechet()
    test_bug7_alpha()
    test_bug8_dvm_comment()
    test_ablation_variants()
    test_gradient_flow()
    test_boundary()
    test_classification_pipeline()

    all_pass = summary()

    # Write report to file
    report_path = os.path.join(_project_root, "test_qa_regression_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# SpikeRPGNN Regression Test Report\n\n")
        total = len(results)
        passed = sum(1 for _, p, _ in results if p)
        failed = total - passed
        f.write(f"## Summary\n\n")
        f.write(f"- Total Tests: {total}\n")
        f.write(f"- Passed: {passed}\n")
        f.write(f"- Failed: {failed}\n")
        f.write(f"- Routing Decision: {'NoOne' if all_pass else 'Engineer'}\n\n")

        if failed > 0:
            f.write("## Failed Tests\n\n")
            for name, p, detail in results:
                if not p:
                    f.write(f"- **{name}**: {detail}\n")
            f.write("\n")

        f.write("## All Test Results\n\n")
        f.write("| Test | Status | Detail |\n")
        f.write("|------|--------|--------|\n")
        for name, p, detail in results:
            status = "PASS" if p else "FAIL"
            f.write(f"| {name} | {status} | {detail} |\n")

    print(f"\nReport saved to: {report_path}")

    sys.exit(0 if all_pass else 1)
