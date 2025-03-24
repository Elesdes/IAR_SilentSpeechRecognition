import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple, Callable, Dict
import torch
from torch.nn import Module
from torch.utils.data import DataLoader


def plot_timeseries_outputs(
    input: torch.Tensor,
    output: torch.Tensor,
    sample_idx: int = 0,
    title: str = None,
    xlabel: str = "Timesteps",
    figure_size: Tuple[int, int] = (8, 3),
) -> None:
    """
    Plotting function for time-series data from recurrent models (e.g., LTC, RNN, Seq2Seq).

    Expects 'input' and 'output' in either [time_steps, batch, dims] or [batch, time_steps, dims] format.
    If the first dimension equals the batch size, it automatically transposes the tensors.

    For 'output':
      - If output is 2D (shape: [time_steps, batch]), a singleton dimension is added.
      - If output is 3D (shape: [time_steps, batch, num_neurons]), then the function plots all neuron outputs
        on a single subplot (with a common y-axis scale).
      - If output is 4D (shape: [time_steps, batch, d1, d2]), the last two dimensions are flattened.

    The function extracts a single sample (specified by sample_idx) and:
      - Plots each input dimension in its own subplot.
      - Plots all neuron outputs together on one subplot, with a common y-axis set by the global min/max.
      - Changes the legend to indicate which color corresponds to which neuron.

    Parameters:
        input (torch.Tensor): Input tensor, shape [time_steps, batch, dims] or [batch, time_steps, dims].
        output (torch.Tensor): Output tensor, shape [time_steps, batch] or [time_steps, batch, num_neurons]
            or [time_steps, batch, d1, d2] (which will be flattened) or in [batch, time_steps, ...] format.
        sample_idx (int): Index of the sample in the batch to plot.
        title (str): Overall title for the plot.
        xlabel (str): Label for the x-axis.
        figure_size (Tuple[int, int]): Base figure size (width, height) for each subplot.
    """
    # Transpose 'input' if needed
    if input.shape[0] < input.shape[1]:
        input = input.transpose(0, 1)

    # Process 'output'
    if output.ndim == 2:
        if output.shape[0] < output.shape[1]:
            output = output.transpose(0, 1)
        output = output.unsqueeze(-1)
    elif output.ndim == 3:
        if output.shape[0] < output.shape[1]:
            output = output.transpose(0, 1)
    elif output.ndim == 4:
        if output.shape[0] < output.shape[1]:
            output = output.transpose(0, 1)
        t, b, d1, d2 = output.shape
        output = output.view(t, b, d1 * d2)
    else:
        raise ValueError("Unexpected output dimensions; expected 2D, 3D, or 4D tensor.")

    time_steps, _, input_dim = input.shape
    num_neurons = output.shape[2]

    # Extract the sample from the batch
    sample_input = input[:, sample_idx, :].detach().cpu().numpy()
    sample_output = output[:, sample_idx, :].detach().cpu().numpy()

    # Create subplots: one for each input dimension + one for all neuron outputs
    total_plots = input_dim + 1
    fig, axs = plt.subplots(
        total_plots,
        1,
        figsize=(figure_size[0], figure_size[1] * total_plots),
        sharex=True,
    )
    if title is not None:
        fig.suptitle(title, fontsize=16)

    # Plot each input dimension in its own subplot
    for i in range(input_dim):
        axs[i].plot(range(time_steps), sample_input[:, i])
        axs[i].set_ylabel(
            f"Input Signal (a.u.)"
            if input_dim == 1
            else f"Input Signal Dim {i+1} (a.u.)"
        )
        axs[i].grid(True)

    # Plot all neuron outputs on a single subplot.
    ax_out = axs[-1]
    # Use a colormap to assign distinct colors for each neuron.
    colors = plt.cm.tab10(np.linspace(0, 1, num_neurons))
    for j in range(num_neurons):
        ax_out.plot(
            range(time_steps),
            sample_output[:, j],
            color=colors[j],
            label=f"Neuron Output {j+1}" if num_neurons > 1 else None,
        )

    # Set the y-axis to a common scale based on the global min and max of the neuron outputs.
    global_min = sample_output.min()
    global_max = sample_output.max()
    ax_out.set_ylim(global_min, global_max)

    ax_out.set_ylabel("Membrane Potential (mV)")
    ax_out.set_xlabel(xlabel)
    ax_out.grid(True)
    if num_neurons > 1:
        ax_out.legend()

    plt.tight_layout()
    plt.show()


def plot_multistep_evaluation(
    model: Module,
    data: DataLoader,
    sample_idx: int = 0,
    dims: Optional[List[str]] = None,
    title: str = "Model Inference Plot",
    xlabel: str = "Time step",
    figure_size: Tuple[int, int] = (10, 15),
    metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
) -> None:
    """
    Run model inference on a sample from the given DataLoader, plot the input, target,
    and predicted time series for each dimension, and display computed metrics on each subplot.
    Also, measure and display the inference speed.

    The function computes several common metrics for evaluation:
        - RMSE: Root Mean Square Error.
        - MAE: Mean Absolute Error.
        - MSE: Mean Squared Error.
        - MAPE: Mean Absolute Percentage Error.
        - R2: Coefficient of Determination.
        - Corr: Pearson Correlation Coefficient.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to perform inference.
    - data (torch.utils.data.DataLoader): DataLoader providing batches of data,
        where each batch is expected to be a tuple (inputs, targets).
    - sample_idx (int): Index of the sample in the batch to plot. Default is 0.
    - dims (List[str], optional): List of dimension names for labeling the plots.
        If None or if the length does not match the number of dimensions in the data,
        defaults to ["Dim 1", "Dim 2", ..., "Dim N"].
    - title (str): Title of the plot. Default is "Model Inference Plot".
    - xlabel (str): Label for the x-axis. Default is "Time step".
    - figure_size (Tuple[int, int]): Size of the figure (width, height). Default is (10, 15).
    - metrics (Dict[str, Callable[[np.ndarray, np.ndarray], float]], optional):
        A dictionary where keys are metric names and values are functions that take
        (target, prediction) numpy arrays for a single dimension and return a float.
        If None, defaults to the following metrics:
            {
                'RMSE': sqrt(mean((t - p)**2)),
                'MAE': mean(|t - p|),
                'MSE': mean((t - p)**2),
                'MAPE': mean(|(t - p)/t|)*100,
                'R2': 1 - (sum((t - p)**2) / sum((t - mean(t))**2)),
                'Corr': Pearson correlation coefficient between t and p
            }

    Returns:
    - None. Displays a plot comparing input, target, and prediction time series for each dimension,
      with metrics and inference speed displayed.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default metrics if none provided
    if metrics is None:
        metrics = {
            "RMSE": lambda t, p: np.sqrt(np.mean((t - p) ** 2)),
            "MAE": lambda t, p: np.mean(np.abs(t - p)),
            "MSE": lambda t, p: np.mean((t - p) ** 2),
            "MAPE": lambda t, p: np.mean(np.abs((t - p) / (t + 1e-8))) * 100,
            "R2": lambda t, p: 1
            - (np.sum((t - p) ** 2) / (np.sum((t - np.mean(t)) ** 2) + 1e-8)),
            "Corr": lambda t, p: (
                float("nan") if len(t) <= 1 else np.corrcoef(t, p)[0, 1]
            ),
        }

    # Fetch a single batch and run inference while timing the process
    with torch.no_grad():
        batch = next(iter(data))
        inputs, targets = batch
        inputs = inputs.float().to(device)
        targets = targets.float().to(device)
        start_time = time.perf_counter()
        try:
            predictions = model(inputs)
        except TypeError:  # To handle Seq2Seq models that require 'y'
            predictions = model(inputs, targets, teacher_forcing_ratio=0.0)
        inference_time = time.perf_counter() - start_time

    avg_time_per_sample = inference_time / inputs.size(0)

    # Select the sample from the batch
    sample_input = inputs[sample_idx].cpu().numpy()  # shape: [time_steps, dims]
    sample_target = targets[sample_idx].cpu().numpy()  # shape: [time_steps, dims]
    sample_pred = predictions[sample_idx].cpu().numpy()  # shape: [time_steps, dims]

    num_timesteps, num_dims = sample_input.shape
    if dims is None or len(dims) != num_dims:
        dims = [f"Dim {i+1}" for i in range(num_dims)]

    fig, axs = plt.subplots(num_dims, 1, figsize=figure_size, sharex=True)
    timesteps = range(num_timesteps)

    # Ensure axs is iterable even when there's only one subplot
    if num_dims == 1:
        axs = [axs]

    for i in range(num_dims):
        # For the first subplot, include labels so legend can be generated
        if i == 0:
            axs[i].plot(timesteps, sample_input[:, i], color="blue", label="Input")
            axs[i].plot(timesteps, sample_target[:, i], color="orange", label="Target")
            axs[i].plot(timesteps, sample_pred[:, i], color="green", label="Predicted")
        else:
            # For other subplots, just use the same colors (no labels)
            axs[i].plot(timesteps, sample_input[:, i], color="blue")
            axs[i].plot(timesteps, sample_target[:, i], color="orange")
            axs[i].plot(timesteps, sample_pred[:, i], color="green")

        axs[i].set_ylabel(f"{dims[i]} value")
        axs[i].grid(True)

        # Compute and display metrics on each subplot
        metric_texts = []
        for metric_name, metric_func in metrics.items():
            value = metric_func(sample_target[:, i], sample_pred[:, i])
            metric_texts.append(f"{metric_name}: {value:.2f}")
        metric_text = ", ".join(metric_texts)

        axs[i].text(
            0.05,
            0.95,
            metric_text,
            transform=axs[i].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.6),
        )

    # Create a single legend from the handles in the first subplot
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,  # number of legend columns
        bbox_to_anchor=(0.5, 0.98),
    )

    axs[-1].set_xlabel(xlabel)
    updated_title = (
        f"{title} (Inference: {inference_time:.3f} sec, "
        f"{avg_time_per_sample * 1000:.2f} ms/sample)"
    )
    plt.suptitle(updated_title, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space at top for legend/title
    plt.show()
