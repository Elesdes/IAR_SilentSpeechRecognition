import numpy as np
import matplotlib.pyplot as plt


class FibonacciOscillator:
    """
    A class to represent a Fibonacci oscillator defined by the generalized
    Fibonacci sequence in its energy spectrum and the commutation relations
    for creation (a^dagger) and annihilation (a) operators.

    The key formula for the (generalized) 'number operator' is:
        [n] = (q1^(2n) - q2^(2n)) / (q1^2 - q2^2),

    where q1 and q2 are the base Fibonacci parameters (> 0).
    """

    def __init__(self, q1: float = 1.1, q2: float = 0.9) -> None:
        """
        Initialize the Fibonacci oscillator with parameters q1 and q2.

        Parameters
        ----------
        q1 : float
            One of the base Fibonacci parameters (default 1.1).
        q2 : float
            The other base Fibonacci parameter (default 0.9).
        """
        self.q1 = q1
        self.q2 = q2

    def number_operator(self, n: int) -> float:
        """
        Compute the generalized Fibonacci number [n] for this oscillator.

        The formula is:
            [n] = (q1^(2n) - q2^(2n)) / (q1^2 - q2^2)

        Parameters
        ----------
        n : int
            Index for the generalized Fibonacci sequence.

        Returns
        -------
        float
            The value of the generalized Fibonacci 'number operator' at index n.
        """
        numerator = self.q1 ** (2 * n) - self.q2 ** (2 * n)
        denominator = self.q1**2 - self.q2**2
        return numerator / denominator

    def energy_levels(self, n_values: np.ndarray) -> np.ndarray:
        """
        Compute the list of energy levels [n] for a set of n-values.

        Parameters
        ----------
        n_values : list or array_like of int
            The sequence of integer indices at which to compute the energy.

        Returns
        -------
        list of float
            The corresponding [n] values.
        """
        return [self.number_operator(n) for n in n_values]

    def plot_energy_levels(
        self, n_min: int = 0, n_max: int = 10, show: bool = True
    ) -> plt.Figure:
        """
        Plot the discrete energy levels [n] for n from n_min to n_max.

        Parameters
        ----------
        n_min : int, optional
            The starting index of n (default 0).
        n_max : int, optional
            The ending index of n (inclusive, default 10).
        show : bool, optional
            If True, display the plot immediately. If False, you can further
            modify the figure before calling plt.show() yourself.

        Returns
        -------
        fig, ax : Matplotlib Figure and Axes
            For further customization if desired.
        """
        n_values = np.arange(n_min, n_max + 1)
        energies = self.energy_levels(n_values)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(n_values, energies, "o-", label="[n] Energy Levels")
        ax.set_xlabel("n")
        ax.set_ylabel("[n]")
        ax.set_title(f"Fibonacci Oscillator Spectrum\n(q1={self.q1}, q2={self.q2})")
        ax.grid(True)
        ax.legend()

        if show:
            plt.show()

        return fig, ax
