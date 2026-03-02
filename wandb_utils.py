from typing import Any, Dict, Optional

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


def init_wandb(config: Dict[str, Any], default_run_name: Optional[str] = None) -> bool:
    """Initialize W&B from config. Returns True when an active run exists."""
    if wandb is None:
        print("Warning: wandb not installed. Skipping wandb initialization.")
        return False

    project = config.get("wandb_project")
    if not project:
        return False

    run_name = config.get("wandb_run_name", default_run_name)
    try:
        wandb.init(project=project, name=run_name, config=config)
    except Exception as exc:
        print(f"Warning: Failed to initialize wandb: {exc}")
        return False
    return wandb.run is not None


def is_wandb_active() -> bool:
    return wandb is not None and wandb.run is not None


def log_wandb(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Safe W&B logging helper (no-op when wandb is unavailable/inactive)."""
    if not is_wandb_active() or not metrics:
        return
    if step is None:
        wandb.log(metrics)
    else:
        wandb.log(metrics, step=step)


def finish_wandb() -> None:
    if is_wandb_active():
        wandb.finish()
