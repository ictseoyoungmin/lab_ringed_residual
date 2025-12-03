from pathlib import Path

LOG_ROOT = Path("./result/logs")


def ensure_log_dir(dataset: str, model: str) -> Path:
    """Return the log directory for the dataset/model pair, creating it if missing."""
    log_dir = LOG_ROOT / dataset / model
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def build_checkpoint_path(dataset: str, model: str, epoch: int, val_dice: float, train_loss: float) -> Path:
    """
    Build a consistent checkpoint path under the log directory.

    The ``epoch`` argument should be the human-readable epoch number (starting at 1),
    so when resuming from a previous checkpoint pass the already completed epoch count
    plus the current zero-based loop index + 1.
    """
    log_dir = ensure_log_dir(dataset, model)
    filename = f"{model}-[val_dice]-{val_dice:.4f}-[train_loss]-{train_loss:.4f}-ep{epoch}.pkl"
    return log_dir / filename


def build_plot_path(dataset: str, model: str, lr: float, suffix: str = None) -> Path:
    """Return the path for a training curve plot stored alongside checkpoints."""
    log_dir = ensure_log_dir(dataset, model)
    suffix = suffix or "Training Process"
    return log_dir / f"{suffix} for lr-{lr}.png"
