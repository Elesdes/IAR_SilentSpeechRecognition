import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum


class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2


class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2


class LTCCell(nn.Module):
    """
    A PyTorch Module implementing the Liquid Time-Constant (LTC) model.
    It's a conversion from the original code which was written using TensorFlow (https://github.com/raminmh/liquid_time_constant_networks/blob/master/experiments_with_ltcs/ltc_model.py).
    This includes:
      - Parameter initialization (weights, leak conductances, etc.)
      - Various ODE solvers (Semi-Implicit, Explicit, Runge-Kutta)
      - Utilities for parameter constraints and exporting weights
    """

    def __init__(
        self,
        num_units: int,
        input_size: int,
        mapping_type: MappingType = MappingType.Affine,
        solver: ODESolver = ODESolver.SemiImplicit,
        ode_solver_unfolds: int = 6,
        w_init_min: float = 0.01,
        w_init_max: float = 1.0,
        cm_init_min: float = 0.5,
        cm_init_max: float = 0.5,
        gleak_init_min: float = 1.0,
        gleak_init_max: float = 1.0,
        erev_init_factor: float = 1.0,
        fix_cm: float | None = None,
        fix_gleak: float | None = None,
        fix_vleak: float | None = None,
    ):
        """
        Args:
            num_units (int): Number of hidden units (neurons).
            input_size (int): Dimensionality of the input.
            mapping_type (MappingType): Whether inputs are used as-is, multiplied by a weight, or multiplied+added bias.
            solver (ODESolver): Which ODE solver to use (SemiImplicit, Explicit, RungeKutta).
            ode_solver_unfolds (int): Number of small ODE steps to perform per forward pass.
            w_init_min, w_init_max (float): Range for random uniform weight initialization.
            cm_init_min, cm_init_max (float): Range for random uniform membrane capacitance init.
            gleak_init_min, gleak_init_max (float): Range for random uniform leak conductance init.
            erev_init_factor (float): Multiplier for random initial E_rev values.
            fix_cm, fix_gleak, fix_vleak (float or None): If set, these parameters are fixed to this value.
        """
        super(LTCCell, self).__init__()

        self.num_units = num_units
        self.input_size = input_size
        self.mapping_type = mapping_type
        self.solver = solver
        self.ode_solver_unfolds = ode_solver_unfolds

        # Initialization hyperparameters
        self.w_init_min = w_init_min
        self.w_init_max = w_init_max
        self.cm_init_min = cm_init_min
        self.cm_init_max = cm_init_max
        self.gleak_init_min = gleak_init_min
        self.gleak_init_max = gleak_init_max
        self.erev_init_factor = erev_init_factor

        # Optionally fix some parameters
        self.fix_cm = fix_cm
        self.fix_gleak = fix_gleak
        self.fix_vleak = fix_vleak

        # Parameter clipping bounds
        self.w_min_value = 1e-5
        self.w_max_value = 1e3
        self.gleak_min_value = 1e-5
        self.gleak_max_value = 1e3
        self.cm_t_min_value = 1e-6
        self.cm_t_max_value = 1e3

        # Create parameters
        self._build_parameters()

    def _build_parameters(self):
        """
        Create and initialize all trainable parameters in the LTC model.
        """

        # Input transformation
        # For Linear/Affine, we have an input scaling factor
        if self.mapping_type in [MappingType.Linear, MappingType.Affine]:
            self.input_w = nn.Parameter(torch.ones(self.input_size))
        else:
            self.input_w = None

        # For Affine, we also have an input bias
        if self.mapping_type == MappingType.Affine:
            self.input_b = nn.Parameter(torch.zeros(self.input_size))
        else:
            self.input_b = None

        # Sensory (input->hidden) parameters
        # mu & sigma
        self.sensory_mu = nn.Parameter(
            torch.rand(self.input_size, self.num_units).uniform_(0.3, 0.8)
        )
        self.sensory_sigma = nn.Parameter(
            torch.rand(self.input_size, self.num_units).uniform_(3.0, 8.0)
        )

        # W
        sensory_W_init = torch.Tensor(self.input_size, self.num_units)
        sensory_W_init.uniform_(self.w_init_min, self.w_init_max)
        self.sensory_W = nn.Parameter(sensory_W_init)

        # E_rev
        # Initialize to +1 or -1, multiplied by factor
        sensory_erev_init = (
            2.0 * torch.randint(low=0, high=2, size=(self.input_size, self.num_units))
            - 1.0
        )
        self.sensory_erev = nn.Parameter(
            sensory_erev_init.float() * self.erev_init_factor
        )

        # Recurrent (hidden->hidden) parameters
        # mu & sigma
        self.mu = nn.Parameter(
            torch.rand(self.num_units, self.num_units).uniform_(0.3, 0.8)
        )
        self.sigma = nn.Parameter(
            torch.rand(self.num_units, self.num_units).uniform_(3.0, 8.0)
        )

        # W
        W_init = torch.Tensor(self.num_units, self.num_units)
        W_init.uniform_(self.w_init_min, self.w_init_max)
        self.W = nn.Parameter(W_init)

        # E_rev
        erev_init = (
            2.0 * torch.randint(low=0, high=2, size=(self.num_units, self.num_units))
            - 1.0
        )
        self.erev = nn.Parameter(erev_init.float() * self.erev_init_factor)

        # Leak parameters
        if self.fix_vleak is None:
            self.vleak = nn.Parameter(torch.rand(self.num_units).uniform_(-0.2, 0.2))
        else:
            vleak_value = torch.tensor(self.fix_vleak, dtype=torch.float32).clone()
            self.vleak = nn.Parameter(vleak_value, requires_grad=False)

        if self.fix_gleak is None:
            if self.gleak_init_max > self.gleak_init_min:
                gleak_init = torch.rand(self.num_units).uniform_(
                    self.gleak_init_min, self.gleak_init_max
                )
            else:
                gleak_init = torch.full((self.num_units,), self.gleak_init_min)
            self.gleak = nn.Parameter(gleak_init)
        else:
            gleak_value = torch.tensor(self.fix_gleak, dtype=torch.float32).clone()
            self.gleak = nn.Parameter(gleak_value, requires_grad=False)

        if self.fix_cm is None:
            if self.cm_init_max > self.cm_init_min:
                cm_init = torch.rand(self.num_units).uniform_(
                    self.cm_init_min, self.cm_init_max
                )
            else:
                cm_init = torch.full((self.num_units,), self.cm_init_min)
            self.cm_t = nn.Parameter(cm_init)
        else:
            cm_value = torch.tensor(self.fix_cm, dtype=torch.float32).clone()
            self.cm_t = nn.Parameter(cm_value, requires_grad=False)

    def forward(self, inputs: torch.Tensor, state: torch.Tensor):
        """
        Perform one forward step of LTC.

        Args:
            inputs (torch.Tensor): Current input [batch_size, input_dim].
            state (torch.Tensor): Previous hidden state [batch_size, num_units].

        Returns:
            outputs (torch.Tensor): Current output (same shape as state).
            next_state (torch.Tensor): Next hidden state.
        """
        # Map inputs if needed
        inputs = self._map_inputs(inputs)

        # Choose ODE solver
        match self.solver:
            case ODESolver.Explicit:
                next_state = self._ode_step_explicit(inputs, state)
            case ODESolver.SemiImplicit:
                next_state = self._ode_step_semimplicit(inputs, state)
            case ODESolver.RungeKutta:
                next_state = self._ode_step_runge_kutta(inputs, state)
            case _:
                raise ValueError("Unknown ODE solver.")

        # In many RNNs, output == next_state
        outputs = next_state
        return outputs, next_state

    def _map_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Optionally applies a linear or affine transform to the inputs.
        """
        if self.mapping_type in [MappingType.Linear, MappingType.Affine]:
            # Multiply by learnable weight
            # input_w has shape [input_dim], so we need broadcast
            inputs = inputs * self.input_w

        if self.mapping_type == MappingType.Affine:
            # Add a learnable bias
            inputs = inputs + self.input_b

        return inputs

    def _ode_step_semimplicit(
        self, inputs: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Semi-implicit (hybrid Euler) method unrolled for self.ode_solver_unfolds steps.
        """
        v_pre = state

        # Precompute sensory activations
        sensory_w_activation = self.sensory_W * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        sensory_rev_activation = sensory_w_activation * self.sensory_erev
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        for _ in range(self.ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev

            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator
            v_pre = numerator / denominator

        return v_pre

    def _ode_step_explicit(
        self, inputs: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Explicit Euler method unrolled for self.ode_solver_unfolds steps.
        """
        v_pre = state

        # Precompute sensory effects
        sensory_w_activation = self.sensory_W * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)

        for _ in range(self.ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            w_reduced_synapse = torch.sum(w_activation, dim=1)

            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation

            sum_in = (
                torch.sum(sensory_in, dim=1)
                - v_pre * w_reduced_synapse
                + torch.sum(synapse_in, dim=1)
                - v_pre * w_reduced_sensory
            )
            f_prime = (1.0 / self.cm_t) * (self.gleak * (self.vleak - v_pre) + sum_in)
            v_pre = v_pre + 0.1 * f_prime

        return v_pre

    def _ode_step_runge_kutta(
        self, inputs: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """
        4th-order Runge-Kutta method (RK4).
        """
        v_pre = state
        h = 0.1
        for _ in range(self.ode_solver_unfolds):
            k1 = h * self._f_prime(inputs, v_pre)
            k2 = h * self._f_prime(inputs, v_pre + 0.5 * k1)
            k3 = h * self._f_prime(inputs, v_pre + 0.5 * k2)
            k4 = h * self._f_prime(inputs, v_pre + k3)

            v_pre = v_pre + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return v_pre

    def _f_prime(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Derivative f'(v) used in the Runge-Kutta solver.
        """
        v_pre = state

        # Sensory input
        sensory_w_activation = self.sensory_W * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)

        # Synaptic input
        w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
        w_reduced_synapse = torch.sum(w_activation, dim=1)

        # Summations
        sensory_in = self.sensory_erev * sensory_w_activation
        synapse_in = self.erev * w_activation
        sum_in = (
            torch.sum(sensory_in, dim=1)
            - v_pre * w_reduced_synapse
            + torch.sum(synapse_in, dim=1)
            - v_pre * w_reduced_sensory
        )

        return (1.0 / self.cm_t) * (self.gleak * (self.vleak - v_pre) + sum_in)

    def _sigmoid(
        self, v_pre: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Element-wise sigmoid activation with shift (mu) and scale (sigma).
        Expects v_pre: [batch_size, num_units], mu/sigma: [*, num_units].
        We'll broadcast shapes appropriately.
        """
        # Insert a dimension so we can broadcast along synapses
        # v_pre => [batch_size, num_units, 1]
        v_reshaped = v_pre.unsqueeze(-1)  # [batch_size, num_units, 1]
        # mu => [*, num_units], so we can do (v_reshaped - mu) if mu has shape [num_units] or [X, num_units]
        # We'll assume shape matches [num_units, num_units] or [input_size, num_units], so broadcast carefully:
        x = sigma * (v_reshaped - mu)
        return torch.sigmoid(x)

    def clip_params(self):
        """
        In-place parameter clipping to ensure values remain within stable ranges.
        Call this after each training step, or periodically, if needed.
        """
        # cm_t
        with torch.no_grad():
            self.cm_t.clamp_(min=self.cm_t_min_value, max=self.cm_t_max_value)

            self.gleak.clamp_(min=self.gleak_min_value, max=self.gleak_max_value)

            self.W.clamp_(min=self.w_min_value, max=self.w_max_value)
            self.sensory_W.clamp_(min=self.w_min_value, max=self.w_max_value)


class LTCModel(nn.Module):
    """
    Multi-step LTCModel that unrolls the LTCCell over the input sequence
    and returns an output at every timestep, matching the target's seq_len.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        ltc_cell_kwargs: dict = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ltc_cell = LTCCell(
            num_units=hidden_size, input_size=input_size, **(ltc_cell_kwargs or {})
        )
        # We'll apply a linear layer to each hidden state to get an output dimension
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, seq_len, input_size]
        Returns:
            predictions: [batch_size, seq_len, output_size]
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        # Hidden state initialization
        h = torch.zeros(batch_size, self.hidden_size, device=device)

        # Collect outputs at each timestep
        outputs = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]  # [batch_size, input_size]
            out, h = self.ltc_cell(x_t, h)
            # Map the hidden state to output_size
            out_mapped = self.fc_out(out)  # [batch_size, output_size]
            outputs.append(out_mapped.unsqueeze(1))  # insert a time dimension

        # Concatenate outputs across time -> [batch_size, seq_len, output_size]
        return torch.cat(outputs, dim=1)
