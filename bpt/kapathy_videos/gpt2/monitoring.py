from dataclasses import asdict
import wandb

from bpt.kapathy_videos.gpt2.model import GPTConfig, RunConfig


def init_wandb(model_config: GPTConfig, run_config: RunConfig): 
    run = wandb.init(
        # Set the wandb entity where your project will be logged (your username)
        entity="bjk95-just-me",  # Changed from "brad" to "bjk95" based on your login
        # Set the wandb project where this run will be logged.
        project="bpt",
        # Track hyperparameters and run metadata.
        config={**asdict(model_config), **asdict(run_config)}
    )