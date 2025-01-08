import numpy as np
from reservoirpy.nodes import Reservoir, Ridge


class ESN:
    """
    A simple Echo State Network (ESN) model using reservoirpy.

    Parameters
    ----------
    input_size : int
        Number of input features.
    output_size : int
        Number of output features.
    units : int
        Number of recurrent units in the reservoir.
    leak_rate : float
        Leaking rate (sometimes called alpha or lr in reservoirpy).
    input_scaling : float
        Scaling factor for input connections.
    bias_scaling : float
        Scaling factor for bias.
    ridge : float
        Regularization parameter for the Ridge regression readout (Tikhonov).
    random_state : int or None
        Seed for reproducible reservoir initialization.
    """

    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        units: int = 100,
        leak_rate: float = 0.3,
        spectral_radius: float = 1.25,
        input_scaling: float = 1.0,
        bias_scaling: float = 0.2,
        ridge: float = 1e-6,
        random_state: int = None,
    ):
        # Create a Reservoir node
        self.reservoir = Reservoir(
            units=units,
            lr=leak_rate,  # `lr` in reservoirpy == leaking rate
            sr=spectral_radius,
            input_scaling=input_scaling,
            bias_scaling=bias_scaling,
            seed=random_state,
        )

        # Create a readout node (Ridge regression by default)
        self.readout = Ridge(ridge=ridge)

        # Compose the full model: data flows from reservoir -> readout
        self.model = self.reservoir >> self.readout

        # Keep track of input/output sizes (just for clarity)
        self.input_size = input_size
        self.output_size = output_size

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the ESN model by fitting the reservoir states to the readout node.

        Parameters
        ----------
        X : np.ndarray
            Training inputs of shape (n_samples, input_size).
        y : np.ndarray
            Training targets of shape (n_samples, output_size).

        Returns
        -------
        The fitted model (self.model).
        """
        # Check shape consistency if you like
        # e.g. ensure X.shape[1] == self.input_size
        self.model = self.model.fit(X, y)
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run the trained ESN model on new data.

        Parameters
        ----------
        X : np.ndarray
            Test inputs of shape (n_samples, input_size).

        Returns
        -------
        y_pred : np.ndarray
            Predictions of shape (n_samples, output_size).
        """
        return self.model.run(X)

    def reset_state(self):
        """
        Reset the reservoir state for a fresh start
        before predicting a new sequence. This is optional and
        depends on time-series usage (online vs. batch).
        """
        self.reservoir.reset()

    def save(self, path: str):
        """
        Save the model to a file.

        Parameters
        ----------
        path : str
            File path to save the model.
        """
        self.model.save(path)
