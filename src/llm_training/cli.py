import click

from llm_training.config import TrainingConfig
from llm_training.trainer import Trainer


def train() -> None:
    """Entry point for training the Transformer language model."""
    config = TrainingConfig(wandb_project="gpt-2-speedrun")

    trainer = Trainer(config)
    trainer.setup_model()
    trainer.setup_dataloader()

    trainer.train()


@click.command()
@click.option("--epochs", default=1000)
def overfit(epochs: int = 1000) -> None:
    """Entry point for testing training setup by overfitting single batch."""
    config = TrainingConfig(wandb_project=None)

    trainer = Trainer(config)
    trainer.setup_model(compile_mode="default")
    trainer.setup_dataloader(overfit=True)

    trainer.overfit(overfitting_epochs=epochs)


if __name__ == "__main__":
    overfit()
