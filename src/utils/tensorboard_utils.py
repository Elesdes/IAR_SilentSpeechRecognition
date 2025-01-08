from tensorboard import program
import subprocess


def launch_tensorboard(log_dir: str) -> None:
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", log_dir])
    url = tb.launch()
    print(f"TensorBoard started at {url}")
    try:
        subprocess.run(["xdg-open", url], check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to open browser. Please manually visit: {url}")
