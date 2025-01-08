from typing import Any, List, Tuple
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.utils.dataset_utils import TimeSeriesDataset


class LorenzOscillator:
    """
    A class to simulate and preprocess the Lorenz oscillator defined by:
        dx/dt = sigma * (y - x)
        dy/dt = rho * x - y - x * z
        dz/dt = x * y - beta * z

    Attributes
    ----------
    sigma : float
        The 'Prandtl' number (assumed > 0).
    rho : float
        The ratio of the Rayleigh number to its critical value (assumed > 0).
    beta : float
        Another positive parameter in the system.

    Methods
    -------
    derivatives(t, state)
        Returns the vector [dx/dt, dy/dt, dz/dt] given current time t and state.
    fixed_points()
        Returns the list of fixed points.
    solve_trajectory(initial_state, t_span=(0, 40), max_step=0.01)
        Integrates the system from an initial state over a given time interval.
    plot_trajectory(initial_state, t_span=(0, 40), max_step=0.01, show=True)
        Integrates the system and plots the 3D trajectory (x, y, z).
    preprocess_and_create_dataset(initial_state, t_span, max_step, input_length, target_length)
        Preprocesses the trajectory data and creates a PyTorch dataset.
    split_and_create_loaders(dataset, batch_size, train_ratio=0.8)
        Splits the dataset into training and testing sets and returns data loaders.
    """

    def __init__(
        self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0
    ) -> None:
        """
        Initialize the Lorenz system with parameters sigma, rho, and beta.
        Defaults are the classical values that produce the strange attractor:
            sigma = 10, rho = 28, beta = 8/3.
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Calculate the time derivatives (dx/dt, dy/dt, dz/dt)
        of the Lorenz system at time t for a given state = (x, y, z).

        Parameters
        ----------
        t : float
            Current time (not used directly, but required by ODE solver).
        state : array_like
            The current values of the variables (x, y, z).

        Returns
        -------
        numpy.ndarray
            Derivatives [dx/dt, dy/dt, dz/dt].
        """
        x, y, z = state
        dxdt = self.sigma * (y - x)
        dydt = self.rho * x - y - x * z
        dzdt = x * y - self.beta * z
        return np.array([dxdt, dydt, dzdt])

    def fixed_points(self) -> List[Tuple[float, float, float]]:
        """
        Compute the fixed points of the Lorenz system (where dx/dt = dy/dt = dz/dt = 0).

        Returns
        -------
        list of tuples
            The fixed points (x, y, z).

        Notes
        -----
        - (0, 0, 0) exists for all parameter values.
        - If rho > 1, two additional symmetric points exist:
              (+/- sqrt(beta * (rho - 1)), +/- sqrt(beta * (rho - 1)), rho - 1).
        """
        points = [(0.0, 0.0, 0.0)]  # Always the origin

        if self.rho > 1:
            val = self.beta * (self.rho - 1)
            r = np.sqrt(val)
            # Symmetric fixed points
            points.append((-r, -r, self.rho - 1))
            points.append((+r, +r, self.rho - 1))

        return points

    def solve_trajectory(
        self,
        *,
        initial_state: np.ndarray,
        t_span: Tuple[float, float] = (0, 40),
        max_step: float = 0.01,
    ) -> solve_ivp:
        """
        Solve the Lorenz system given an initial state over a time interval.

        Parameters
        ----------
        initial_state : array_like
            The initial state [x0, y0, z0].
        t_span : tuple, optional
            The time interval (start, end) for integration. Default is (0, 40).
        max_step : float, optional
            The maximum time step for the solver. Smaller steps can capture
            chaotic behavior more accurately. Default is 0.01.

        Returns
        -------
        sol : OdeResult
            The result object returned by solve_ivp, containing:
            - sol.t : array of time points
            - sol.y : array of shape (3, len(t)) with the [x, y, z] solutions
        """
        return solve_ivp(
            fun=self.derivatives, t_span=t_span, y0=initial_state, max_step=max_step
        )

    def plot_trajectory(
        self,
        *,
        initial_state: np.ndarray,
        t_span: Tuple[float, float] = (0, 40),
        max_step: float = 0.01,
        show: bool = True,
    ):
        """
        Solve and plot the Lorenz trajectory in 3D.

        Parameters
        ----------
        initial_state : array_like
            Initial condition [x0, y0, z0].
        t_span : tuple, optional
            Time interval (start, end) for integration (default is (0, 40)).
        max_step : float, optional
            Max time step for the solver (default is 0.01).
        show : bool, optional
            If True, calls plt.show() to display the plot (default True).
            Set to False if you want to modify the figure before showing.

        Returns
        -------
        fig, ax : matplotlib Figure and 3D Axes
            You can modify them further if needed.
        """
        sol = self.solve_trajectory(
            initial_state=initial_state, t_span=t_span, max_step=max_step
        )

        x_data, y_data, z_data = sol.y
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x_data, y_data, z_data, color="tab:blue", lw=1.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(
            "Lorenz Attractor (sigma={}, rho={}, beta={})".format(
                self.sigma, self.rho, self.beta
            )
        )
        if show:
            plt.show()
        return fig, ax

    def preprocess_and_create_dataset(
        self,
        initial_state: List[float] = [1.0, 1.0, 1.0],
        t_span: Tuple[int, int] = (0, 40),
        max_step: float = 1e-2,
        input_length: int = 50,
        target_length: int = 1,
    ) -> TimeSeriesDataset:
        """
        Preprocess the Lorenz system data and create a PyTorch dataset.

        Parameters
        ----------
        initial_state : array_like
            The initial state [x0, y0, z0].
        t_span : tuple
            The time interval (start, end) for integration.
        max_step : float
            Maximum time step for the solver.
        input_length : int
            Number of timesteps in the input sequence.
        target_length : int
            Number of timesteps in the target sequence.

        Returns
        -------
        dataset : TimeSeriesDataset
            PyTorch dataset containing normalized input and target sequences.
        """
        sol = self.solve_trajectory(
            initial_state=initial_state, t_span=t_span, max_step=max_step
        )
        data = np.vstack([sol.y[0], sol.y[1], sol.y[2]]).T
        scaler = MinMaxScaler()
        data_norm = scaler.fit_transform(data)
        return TimeSeriesDataset(data_norm, input_length, target_length)
