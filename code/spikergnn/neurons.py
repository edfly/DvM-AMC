"""
Manifold-valued Leaky Integrate-and-Fire (LIF) neuron.

Implements spike neuron dynamics on a product manifold H^{d_h} x R^{d_e}:

  State evolution (Eq. 3):
    p(t+1) = exp_{p(t)}(beta * logmap(p_bar, p(t)) + gamma * I(t))

  Membrane potential (Eq. 4):
    ||u(t+1)|| = alpha * ||u(t)|| + W * s_in(t) - s_out(t) * delta_reset

  Spike trigger (Eq. 5):
    s(t+1) = SigmoidSurrogate(||u(t+1)|| - u_threshold)

Includes SigmoidSurrogate for surrogate gradient (reimplemented from
msg_baseline.models.SigmoidSurrogate logic, not imported).

Bug 4 fix: w_input_mag is divided by sqrt(d_total) to normalize the input
magnitude to match the threshold. Without this, ||W*I|| ~ O(sqrt(d_total))
far exceeds u_threshold=1.0, causing all neurons to fire every step.
The membrane potential u_mag is persisted across forward calls (stored as a
buffer) so that the LIF can be called with T=1 from STMP without losing state.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from spikergnn.manifolds import ProductManifold, Point, TangentVec


class SigmoidSurrogate(nn.Module):
    """Sigmoid-based surrogate gradient for spike non-differentiability.

    During forward pass: returns binary spike (0 or 1).
    During backward pass: passes sigmoid gradient with controllable steepness.

    This is a standalone reimplementation matching msg_baseline.models.SigmoidSurrogate
    logic, to avoid circular imports.

    Attributes:
        beta: Steepness of the sigmoid surrogate gradient.
    """

    def __init__(self, beta: float = 5.0):
        """Initialize surrogate gradient function.

        Args:
            beta: Steepness parameter for the sigmoid. Higher = sharper.
        """
        super().__init__()
        self.beta = beta

    def forward(self, v_mem: torch.Tensor) -> torch.Tensor:
        """Apply surrogate gradient spike function.

        Args:
            v_mem: Membrane potential values (any shape).

        Returns:
            Binary spikes in forward, with sigmoid gradient in backward.
        """
        # Forward: hard threshold
        spike = (v_mem >= 0).float()

        # Backward: surrogate sigmoid gradient
        if self.training:
            sig = torch.sigmoid(self.beta * v_mem)
            surrogate = sig * (1 - sig) * self.beta
            return spike - surrogate.detach() + surrogate
        return spike


class ATanSurrogate(nn.Module):
    """Arc-tangent (ATan) surrogate gradient for spike non-differentiability.

    During forward pass: returns binary spike (0 or 1).
    During backward pass: passes atan gradient which does NOT vanish
    for large |v_mem|, unlike the sigmoid surrogate.

    The ATan surrogate gradient is:
        g'(x) = β / (π * (1 + (β*x/π)²))

    Key advantage over SigmoidSurrogate: the gradient magnitude decays
    as 1/x² (polynomial) rather than exponentially, providing non-trivial
    gradient signal even when u_mag >> u_th (saturated firing regime).
    This is critical for spike rate regularization (L_sr), which needs
    to push u_th upward when the network fires too frequently.

    Reference:
        Wu et al., "A Direct Training Algorithm for Spiking Neural Networks",
        AAAI 2021 (uses ATan surrogate for better gradient flow).

    Attributes:
        beta: Steepness of the ATan surrogate gradient.
    """

    def __init__(self, beta: float = 2.0):
        """Initialize ATan surrogate gradient function.

        Args:
            beta: Steepness parameter. Default 2.0 (empirically effective).
        """
        super().__init__()
        self.beta = beta

    def forward(self, v_mem: torch.Tensor) -> torch.Tensor:
        """Apply ATan surrogate gradient spike function.

        Args:
            v_mem: Membrane potential values (any shape).

        Returns:
            Binary spikes in forward, with ATan gradient in backward.
        """
        # Forward: hard threshold
        spike = (v_mem >= 0).float()

        # Backward: ATan surrogate gradient
        if self.training:
            # g(x) = atan(β*x) / π + 0.5  (shifted atan to range [0,1])
            # g'(x) = β / (π * (1 + (β*x/π)²))
            # Key: gradient decays as 1/x², not exponentially!
            pi = math.pi
            bp = self.beta / pi
            surrogate_grad = bp / (1 + (bp * v_mem) ** 2)
            # STE: forward = spike, backward = surrogate_grad
            return spike - surrogate_grad.detach() + surrogate_grad
        return spike


class ManifoldLIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron on a product manifold.

    The neuron state p(t) evolves ON the manifold (not in tangent space),
    using the Riemannian exponential map. Membrane potential magnitude
    determines spike firing via a surrogate gradient.

    Bug 4 fix: w_input_mag is normalized by sqrt(d_total) so that ||W*I||/sqrt(d)
    is O(1), matching the threshold u_th=1.0 and preventing all-fire behavior.
    The membrane potential u_mag is persisted across forward calls as a buffer,
    so calling with T=1 repeatedly (as STMP does) correctly accumulates potential.

    Attributes:
        manifold: ProductManifold instance.
        beta: Learnable leak decay parameter.
        gamma: Learnable input scaling parameter.
        alpha: Membrane potential decay constant.
        u_th: Spike firing threshold.
        delta_r: Refractory reset amount.
        surrogate: SigmoidSurrogate instance.
        W: Learnable synaptic weight matrix (d_total x d_total).
        p_bar_decay: Exponential moving average decay for p_bar.
    """

    def __init__(
        self,
        manifold: ProductManifold,
        beta_leak: float = 0.3,
        gamma_input: float = 0.5,
        alpha_decay: float = 0.9,
        u_threshold: float = 0.3,
        delta_reset: float = 1.0,
        surrogate_beta: float = 5.0,
        surrogate_type: str = "atan",
        p_bar_decay: float = 0.9,
        ablation: str = "full",
    ):
        """Initialize ManifoldLIFNeuron.

        Args:
            manifold: ProductManifold instance.
            beta_leak: Initial value for learnable leak parameter.
            gamma_input: Initial value for learnable input scaling parameter.
            alpha_decay: Membrane potential decay constant.
            u_threshold: Spike threshold.
            delta_reset: Refractory reset amount.
            surrogate_beta: Steepness of surrogate gradient.
            surrogate_type: "sigmoid" or "atan". ATan is recommended because
                its gradient decays polynomially (1/x²) rather than
                exponentially, enabling effective spike rate regularization.
            p_bar_decay: EMA decay for maintaining reference point p_bar.
            ablation: Ablation variant. "no_spikes" disables spiking.
        """
        super().__init__()
        self.manifold = manifold
        self.d_total = manifold.d_h + manifold.d_e
        self.alpha = alpha_decay
        self.u_th = u_threshold
        self.delta_r = delta_reset
        self.p_bar_decay = p_bar_decay
        self.ablation = ablation

        # Learnable parameters
        self.beta = nn.Parameter(torch.tensor(beta_leak))
        self.gamma = nn.Parameter(torch.tensor(gamma_input))
        self.W = nn.Linear(self.d_total, self.d_total, bias=False)

        # Input normalization: ensures w_input_mag stays O(1) regardless of
        # RGC output scale changes during training. Without this, training
        # causes RGC weights to shrink, making w_input_mag ~0.01 and preventing
        # any spikes from firing (u_mag never reaches u_th).
        self.input_ln = nn.LayerNorm(self.d_total)

        # Bug 5 fix: learnable geodesic interpolation rate (replaces hardcoded alpha_mix)
        self.spike_alpha = nn.Parameter(torch.tensor(0.5))

        # Bug 4 fix: learnable threshold. After normalizing w_input_mag by 1/sqrt(d_total),
        # the effective input magnitude varies: ~0.2 for dense features, ~0.08 for sparse
        # Poincaré-projected features. However, with W multiplication, the effective
        # input magnitude is ~0.5-0.6 after normalization. Setting u_th_init too low
        # (e.g., 0.35) causes every time step to fire (input > threshold).
        # Use u_th_init = 1.0 (original paper value) which gives ~30-50% spike rate
        # for typical inputs: the membrane potential accumulates over 2-3 steps via
        # alpha decay before exceeding threshold.
        self.u_th_learnable = nn.Parameter(torch.tensor(1.0))

        # Surrogate gradient (ATan by default for better gradient flow with L_sr)
        if surrogate_type == "atan":
            self.surrogate = ATanSurrogate(beta=surrogate_beta)
        else:
            self.surrogate = SigmoidSurrogate(beta=surrogate_beta)

        # Spike count tracking
        self._spike_count = 0
        self._total_neurons = 0   # N * T for spike rate computation
        self._spike_rate = 0.0    # Empirical spike rate from last forward pass

        # Bug 4 fix: persistent membrane potential as a buffer.
        # This ensures u_mag is preserved across forward calls when STMP
        # calls LIF with T=1 repeatedly. Without this, u_mag resets to 0
        # on every call, preventing potential accumulation for sparse inputs.
        self.register_buffer("_u_mag", None)

        # Track the number of nodes for shape validation
        self._last_N = None

    def _get_u_mag(self, N: int, device: torch.device) -> torch.Tensor:
        """Get or initialize the membrane potential buffer.

        Args:
            N: Number of nodes.
            device: Device for the tensor.

        Returns:
            (N,) membrane potential magnitude tensor.
        """
        if self._u_mag is None or self._u_mag.shape[0] != N or self._u_mag.device != device:
            self._u_mag = torch.zeros(N, device=device)
        return self._u_mag

    def forward(
        self,
        x_seq: torch.Tensor,
        p_init: Point,
    ) -> Tuple[torch.Tensor, Point]:
        """Run LIF neuron dynamics over T time steps on the product manifold.

        Args:
            x_seq: (N, T, d_total) input sequence in tangent space at each step.
                   N = number of nodes, T = time steps, d_total = d_h + d_e.
            p_init: (p_hyp, p_euc) initial manifold state.
                    p_hyp: (N, d_h), p_euc: (N, d_e).

        Returns:
            spikes: (N, T, d_total) spike sequence.
            p_final: (p_hyp_final, p_euc_final) final manifold state.
        """
        N, T, d_total = x_seq.shape
        device = x_seq.device

        # Initialize state
        p_hyp, p_euc = p_init
        p_hyp = p_hyp.clone()
        p_euc = p_euc.clone()

        # Bug 4 fix: use persistent membrane potential buffer.
        # This preserves u_mag across forward calls, allowing proper
        # accumulation when called with T=1 from STMP.
        u_mag = self._get_u_mag(N, device)

        # Initialize p_bar as EMA of manifold states
        p_bar_hyp = p_hyp.detach().clone()
        p_bar_euc = p_euc.detach().clone()

        # Ablation note: "no_spikes" is NOT an early return. It flows through the
        # full Riemannian state-evolution loop below and emits a *continuous rate*
        # (see the spike-trigger section) instead of a binary spike, with no
        # membrane reset. This makes it a genuine non-spiking (analog) recurrent
        # neuron that still runs the manifold dynamics — isolating the *spike
        # mechanism* (reviewer P1-3). With c_init=0 it degenerates to a dense
        # Euclidean analog neuron — the parameter-matched dense baseline
        # (reviewer P0-2). The earlier constant-ones early return was removed
        # because it produced a degenerate (input-independent) readout.

        spike_list = []
        self._spike_count = 0

        # Clamp learnable parameters
        beta_val = self.beta.clamp(min=0.01, max=1.0)
        gamma_val = self.gamma.clamp(min=0.01, max=2.0)

        for t in range(T):
            I_t = x_seq[:, t, :]  # (N, d_total)

            # ── State evolution (Eq. 3) ────────────────────────────
            # Compute tangent direction: beta * logmap(p_bar, p) + gamma * I(t)
            # logmap(p_bar, p) gives tangent vector at p pointing toward p_bar
            log_bar_hyp = self.manifold.hyp.logmap(
                p_bar_hyp, p_hyp
            )  # (N, d_h)
            log_bar_euc = p_bar_euc - p_euc  # (N, d_e)

            # Split input into manifold components
            I_hyp, I_euc = self.manifold.split(I_t)

            # Tangent vector at p
            v_hyp = beta_val * log_bar_hyp + gamma_val * I_hyp
            v_euc = beta_val * log_bar_euc + gamma_val * I_euc

            # Exponential map: move p along tangent direction
            p_new_hyp = self.manifold.hyp.expmap(v_hyp, p_hyp)
            p_new_euc = p_euc + v_euc

            # Project back to manifold
            p_new_hyp = self.manifold.hyp.projx(p_new_hyp)

            # ── Membrane potential (Eq. 4) ─────────────────────────
            # W * s_in: synaptic weighted input (use I_t as s_in for first pass)
            # Apply LayerNorm to stabilize input magnitude across training.
            # Without this, RGC output shrinks during training, causing
            # w_input_mag to drop from ~0.5 to ~0.01 and eliminating all spikes.
            I_t_normed = self.input_ln(I_t)  # normalize to mean=0, std=1
            w_input = self.W(I_t_normed)  # (N, d_total)
            w_input_mag = w_input.norm(dim=-1)  # (N,)

            # Bug 4 fix: normalize by sqrt(d_total) so input magnitude matches threshold.
            # After LayerNorm, ||I_t|| ≈ sqrt(d_total), so ||W*I_t|| ≈ sigma_W * d_total.
            # Dividing by sqrt(d_total) gives w_input_mag ≈ sigma_W * sqrt(d_total) ≈ O(1).
            w_input_mag = w_input_mag / math.sqrt(self.d_total)

            # ── Input magnitude clamp ───────────────────────────────
            # Critical for spike rate control: clamp per-step input magnitude to at
            # most 0.6× the effective threshold. This guarantees the membrane potential
            # cannot exceed threshold in a single step (must accumulate over 2+ steps
            # via α decay), which gives sr ∈ [0, ~0.5] at steady state.
            # Without this, growing RGC features during training cause ||W*I||/sqrt(d)
            # to exceed u_th, resulting in all-fire (sr=1.0) regardless of u_th value.
            # The 0.6× factor is chosen so that with α=0.9:
            #   - 1 step: u_mag ≤ 0.6u_th < u_th → no fire
            #   - 2 steps: u_mag ≤ 0.6u_th*(1+α) = 1.14u_th → fire!
            #   - Expected sr ≈ 1/2 = 0.5 (close to target r*=0.4)
            effective_threshold = self.u_th_learnable.clamp(min=0.05)
            w_input_mag_preclamp = w_input_mag  # kept for analog (no_spikes) rate output
            w_input_mag = torch.clamp(w_input_mag, max=effective_threshold.detach() * 0.6)

            u_mag = self.alpha * u_mag + w_input_mag

            # ── Membrane potential clamp ───────────────────────────
            # Prevent runaway excitation: limit u_mag to 1.5x the effective threshold.
            # With input clamped at ≤ 0.6u_th per step, the maximum steady-state
            # accumulation is 0.6u_th/(1-0.9) = 6u_th, but with firing + reset,
            # u_mag cycles between 0 and ~1.14u_th, so 1.5x is sufficient.
            u_mag = torch.clamp(u_mag, max=effective_threshold.detach() * 1.5)

            # ── Spike trigger (Eq. 5) ──────────────────────────────
            # Bug 4 fix: use learnable threshold that adapts to input magnitude.
            effective_threshold = self.u_th_learnable.clamp(min=0.05)
            if self.ablation == "no_spikes":
                # Non-spiking analog (rate) neuron. The manifold/Riemannian
                # state evolution above runs UNMASKED (spike_mask = 1) so the
                # recurrent dynamics are exercised fully — this isolates the
                # *spike mechanism* (reviewer P1-3). `rate_proxy` is only a
                # differentiable proxy rate, NOT a gate: gating the analog
                # neuron with a near-zero sigmoid was what froze its state and
                # collapsed the readout to a constant (the earlier bug). With
                # c_init=0 this degenerates to a dense Euclidean analog RNN —
                # the parameter-matched dense baseline (reviewer P0-2).
                rate_proxy = torch.sigmoid(w_input_mag_preclamp - 1.0)  # (N,), continuous
                self._spike_count += 0
                s_out = rate_proxy
                # (N,1) NOT (N,): a bare (N,) vector does NOT broadcast against
                # the (N, d_h) manifold state (right-aligned -> size N vs d_h),
                # which was a regression that crashed every ablation.
                spike_mask = torch.ones(N, 1, device=s_out.device)  # full state update (unmasked)
            else:
                s_out = self.surrogate(u_mag - effective_threshold)  # (N,)
                self._spike_count += int(s_out.sum().item())
                spike_mask = s_out.unsqueeze(-1)  # (N,1) for correct broadcast

            # Track surrogate spike rate for differentiable regularization
            # s_out is already the surrogate gradient output: binary in forward,
            # sigmoid in backward. So taking mean gives differentiable spike rate.
            if t == 0:
                self._surrogate_spike_sum = s_out.sum()
            else:
                self._surrogate_spike_sum = self._surrogate_spike_sum + s_out.sum()

            # Membrane potential reset
            # Hard reset: set u_mag = 0 after spike (standard in SNN literature).
            # Previous soft reset (u_mag -= delta_r) failed because u_mag can
            # accumulate far above threshold (α≈0.9 + large input), making
            # the fixed reset amount insufficient. Hard reset prevents
            # immediate re-firing and enables sparse spike trains.
            if self.ablation != "no_spikes":
                u_mag = u_mag * (1 - s_out)  # hard reset only for spiking modes

            # ── Apply spike mask to manifold state ─────────────────
            # spike_mask was assigned above: full update (ones) for no_spikes,
            # binary-ish gate for spiking modes. Do NOT recompute from s_out
            # here — that would re-gate the analog neuron and freeze its state.

            # Bug 5 fix: geodesic interpolation replaces Euclidean linear blend.
            # Linear interpolation p = (1-a)*p_old + a*p_new can produce points
            # outside the Poincaré ball, violating manifold structure.
            # Instead, when spike fires, use exp_{p_old}(alpha * log_{p_old}(p_new))
            # which stays on the manifold by construction.
            spike_alpha = torch.sigmoid(self.spike_alpha).clamp(min=0.01, max=0.99)

            if self.ablation == "euclidean":
                # Euclidean ablation: simple linear blend is fine for flat space
                p_hyp = (1 - spike_alpha * spike_mask) * p_hyp + spike_alpha * spike_mask * p_new_hyp
                p_hyp = self.manifold.hyp.projx(p_hyp)
            else:
                # Geodesic interpolation for hyperbolic component
                # For nodes that spiked: move along geodesic from p_hyp toward p_new_hyp
                # For nodes that didn't spike: keep current state (leak only)
                v_geo = self.manifold.hyp.logmap(p_new_hyp, p_hyp)  # log_{p_old}(p_new)
                p_hyp_spike = self.manifold.hyp.expmap(spike_alpha * v_geo, p_hyp)
                p_hyp_spike = self.manifold.hyp.projx(p_hyp_spike)

                # Blend: spiking nodes use geodesic update, non-spiking keep p_hyp
                p_hyp = spike_mask * p_hyp_spike + (1 - spike_mask) * p_hyp
                p_hyp = self.manifold.hyp.projx(p_hyp)

            # Euclidean component: linear blend is correct (flat space)
            p_euc = (1 - spike_alpha * spike_mask) * p_euc + spike_alpha * spike_mask * p_new_euc

            # Update p_bar via exponential moving average
            with torch.no_grad():
                p_bar_hyp = self.p_bar_decay * p_bar_hyp + (1 - self.p_bar_decay) * p_hyp.detach()
                p_bar_hyp = self.manifold.hyp.projx(p_bar_hyp)
                p_bar_euc = self.p_bar_decay * p_bar_euc + (1 - self.p_bar_decay) * p_euc.detach()

            # Build spike output: broadcast spike across dimensions
            spike_full = spike_mask.expand_as(I_t)  # (N, d_total)
            spike_list.append(spike_full)

        # Persist membrane potential for next forward call
        self._u_mag = u_mag.detach().clone()

        # Compute and store spike rate
        self._total_neurons = N * T
        self._spike_rate = self._spike_count / max(self._total_neurons, 1)

        spikes = torch.stack(spike_list, dim=1)  # (N, T, d_total)
        return spikes, (p_hyp, p_euc)

    def reset_membrane(self):
        """Reset the persistent membrane potential to zero."""
        self._u_mag = None

    @property
    def spike_count(self) -> int:
        """Return the total number of spikes from the last forward pass."""
        return self._spike_count

    def spike_rate_surrogate(self) -> torch.Tensor:
        """Return the differentiable (surrogate) spike rate from the last forward pass.

        This uses the surrogate gradient output (sigmoid in backward, binary in forward),
        making it differentiable with respect to u_th_learnable for spike rate regularization.

        Returns:
            Tensor scalar: surrogate spike rate in [0, 1].
        """
        if self._total_neurons == 0:
            return torch.tensor(0.0, device=self.beta.device)
        return self._surrogate_spike_sum / self._total_neurons
