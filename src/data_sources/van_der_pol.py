from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class VanDerPolOscillator:
    """
    A class to simulate the Van der Pol oscillator, which is a nonlinear
    second-order ODE of the form:
        d^2x/dt^2 - epsilon omega_0 (1 - x^2) dx/dt + omega_0^2 x = 0  (free oscillator)

    or (forced oscillator):
        d^2x/dt^2 - epsilon omega_0 (1 - x^2) dx/dt + omega_0^2 x = omega_0^2 * X * cos(omega t)

    where:
      - x(t) is the displacement coordinate
      - omega_0 is the natural (angular) frequency
      - epsilon is the nonlinear damping/gain coefficient
      - (optional) X is the forcing amplitude
      - (optional) omega is the forcing frequency
    """

    def __init__(
        self,
        omega_0: float = 1.0,
        epsilon: float = 0.1,
        forcing_amplitude: float = 0.0,
        forcing_freq: float = 1.0,
    ) -> None:
        """
        Initialize the Van der Pol oscillator with parameters:

        Parameters
        ----------
        omega_0 : float
            Natural frequency of the oscillator (default 1.0).
        epsilon : float
            Nonlinear damping/gain parameter (default 0.1).
        forcing_amplitude : float
            Amplitude of external forcing (default 0.0 => no forcing).
        forcing_freq : float
            Frequency of external forcing (default 1.0).
        """
        self.omega_0 = omega_0
        self.epsilon = epsilon
        self.forcing_amplitude = forcing_amplitude
        self.forcing_freq = forcing_freq

    def derivatives(self, t: float, state: np.ndarray) -> Tuple[float, float]:
        """
        Returns the first-order derivatives for the system in state-space form.

        Here we convert the second-order ODE to a pair of first-order ODEs:
            x' = v
            v' = epsilon omega_0 (1 - x^2) v - omega_0^2 x + [ omega_0^2 * X cos(omega t) if forced ]

        Parameters
        ----------
        t : float
            Current time (used if forcing_amplitude != 0).
        state : array_like
            The current [x, v], where v = dx/dt.

        Returns
        -------
        dxdt, dvdt : tuple of floats
            The time derivatives [dx/dt, dv/dt].
        """
        x, v = state
        # Free oscillator contribution
        dvdt = self.epsilon * self.omega_0 * (1.0 - x**2) * v - self.omega_0**2 * x

        # If forced oscillator, add external driving term
        if self.forcing_amplitude != 0.0:
            dvdt += (
                self.omega_0**2 * self.forcing_amplitude * np.cos(self.forcing_freq * t)
            )

        return [v, dvdt]

    def solve_trajectory(
        self,
        *,
        initial_state: np.ndarray,
        t_span: Tuple[float, float] = (0, 50),
        max_step: float = 0.01,
    ) -> solve_ivp:
        """
        Solve the Van der Pol oscillator from an initial state over a time interval.

        Parameters
        ----------
        initial_state : array_like
            The initial conditions [x0, v0].
        t_span : tuple of floats
            The (start, end) time for integration (default (0, 50)).
        max_step : float
            Maximum step size for the solver (default 0.01).

        Returns
        -------
        sol: OdeResult
            Integration result from solve_ivp. Access .t for time,
            .y for states (shape: 2 x len(t)).
        """
        return solve_ivp(
            fun=self.derivatives,
            t_span=t_span,
            y0=initial_state,
            max_step=max_step,
            dense_output=True,  # for interpolation if needed
        )

    def plot_trajectory(
        self,
        *,
        initial_state: np.ndarray,
        t_span: Tuple[float, float] = (0, 50),
        max_step: float = 0.01,
        show: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """
        Integrate the system and plot:
            - x(t) vs. t
            - the phase portrait: v(t) vs. x(t)

        Parameters
        ----------
        initial_state : array_like
            The initial conditions [x0, v0].
        t_span : tuple, optional
            The (start, end) time for integration (default (0, 50)).
        max_step : float
            Maximum step size for the solver (default 0.01).
        show : bool
            Whether to show the plots immediately (default True).

        Returns
        -------
        (fig, ax1, ax2) : matplotlib Figure and Axes
            For further customization if desired.
        """
        sol = self.solve_trajectory(
            initial_state=initial_state, t_span=t_span, max_step=max_step
        )
        t_vals = sol.t
        x_vals = sol.y[0]
        v_vals = sol.y[1]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        # fig.suptitle("Van der Pol Oscillator")

        # x(t) vs t
        ax1.plot(t_vals, x_vals, "b-", label="x(t)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("x(t)")
        ax1.set_title("Time Evolution")
        ax1.grid(True)
        ax1.legend()

        # Phase portrait: v vs x
        ax2.plot(x_vals, v_vals, "r-", label="Phase Portrait")
        ax2.set_xlabel("x")
        ax2.set_ylabel("v = dx/dt")
        ax2.set_title("Phase Space")
        ax2.grid(True)
        ax2.legend()

        if show:
            plt.tight_layout()
            plt.show()

        return fig, ax1, ax2


if __name__ == "__main__":
    init_state = [1.0, 0.0]  # x0=1, v0=0

    # Free Van der Pol oscillator example
    vdp_free = VanDerPolOscillator(omega_0=1.0, epsilon=0.2, forcing_amplitude=0.0)
    vdp_free.plot_trajectory(initial_state=init_state, t_span=(0, 50))

    # Forced Van der Pol oscillator example
    vdp_forced = VanDerPolOscillator(
        omega_0=1.0, epsilon=0.2, forcing_amplitude=0.5, forcing_freq=1.2
    )
    vdp_forced.plot_trajectory(initial_state=init_state, t_span=(0, 50))
