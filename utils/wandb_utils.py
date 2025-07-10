import wandb


def login_and_get_runs(entity_project: str, filters: dict):
    """Login to wandb and return runs for the given project."""
    _ = wandb.login()
    api = wandb.Api()
    return api.runs(entity_project, filters=filters)
