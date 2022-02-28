import argparse
import pytorch_lightning as pl
import satellighte as sat
import torch
import torchmetrics as tm
import torchvision.transforms as tt
from torch.utils.data import DataLoader


def parse_arguments():
    """
    Parse command line arguments.

    Returns: Parsed arguments
    """
    arg = argparse.ArgumentParser()

    arg.add_argument(
        "--max_epoch",
        type=int,
        default=10,
        help="Maximum number of epochs to train the model",
    )
    arg.add_argument(
        "--device",
        type=int,
        default=1 if torch.cuda.is_available() else 0,
        choices=[0, 1],
        help="GPU device to use",
    )
    arg.add_argument(
        "--arch",
        type=str,
        default=sat.available_archs()[0],
        choices=sat.available_archs(),
        help="Model architecture",
    )
    arg.add_argument(
        "--config",
        type=str,
        default=sat.get_arch_configs(sat.available_archs()[0])[0],
        help="Model configuration",
    )
    arg.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
    )
    arg.add_argument(
        "--data_dir",
        type=str,
        default="satellighte/datas/eurosat",
        help="Path to the dataset",
    )
    arg.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    arg.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size for evaluation",
    )
    arg.add_argument(
        "--num_workers",
        type=int,
        default=3,
        help="Number of workers for data loading",
    )
    arg.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility",
    )

    # Hyperparameters
    arg.add_argument(
        "--criterion",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy"],
        help="Criterion to use for training",
    )
    arg.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="Optimizer to use for training",
    )
    arg.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for training",
    )
    arg.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for training",
    )
    arg.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay for training",
    )
    arg.add_argument(
        "--betas",
        type=tuple,
        default=(0.9, 0.999),
        help="Betas for Adam optimizer",
    )
    arg.add_argument(
        "--eps",
        type=float,
        default=1e-08,
        help="Epsilon for Adam optimizer",
    )
    arg.add_argument(
        "--scheduler",
        type=str,
        default="steplr",
        choices=["steplr", "multisteplr"],
        help="Scheduler to use for training",
    )
    arg.add_argument(
        "--step_size",
        type=float,
        default=4,
        help="Step size for scheduler",
    )
    arg.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma for scheduler",
    )
    arg.add_argument(
        "--milestones",
        type=list,
        default=[500000, 1000000, 1500000],
        help="Milestones for scheduler",
    )
    return arg.parse_args()


def main(args):
    """
    Main function that training the model.

    Args:
        args : Parsed arguments
    """
    # Set seed for reproducibility
    pl.seed_everything(args.seed)

    # Build transforms for training and validation
    # train_transforms = tt.transforms.Compose(
    # tt.transforms.RandomHorizontalFlip(p=0.5),
    # )

    # Load datasets from the sat.dataset module

    train_ds = sat.datasets.EuroSAT(
        root_dir=args.data_dir,
        phase="train",
    )
    val_ds = sat.datasets.EuroSAT(
        root_dir=args.data_dir,
        phase="val",
    )

    # Create a DataLoader for training and validation
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Define hyperparameters
    hparams = {
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "milestones": [500000, 1000000, 1500000],
        "gamma": 0.1,
        "ratio": 10,
        "criterion": args.criterion,
    }

    # Build the model with given architecture and configuration with randomly initialized weights.
    model = sat.Classifier.build(
        args.arch,
        config=args.config,
        hparams=hparams,
        labels=train_ds.classes,
    )

    # Saved model name
    model_save_name = f"{args.arch}_{args.config}_best"

    # Define metrics
    model.add_metric("accuracy", tm.Accuracy())

    # Define checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.save_dir,  # Path to save the model
        verbose=True,  # Whether to print information about the model checkpointing
        filename=model_save_name,  # Filename to save the model
        monitor="metrics/accuracy",  # Quantity to monitor
        save_top_k=1,  # Number of models to keep
        mode="max",  # Save max metric value
    )

    # Define trainer
    trainer = pl.Trainer(
        default_root_dir=".",  # Default path for logs and weights
        accumulate_grad_batches=4,  # Accumulates grads every k batches
        callbacks=[
            checkpoint_callback,
            pl.callbacks.RichProgressBar(),
        ],  # Callback for checkpointing
        gpus=args.device,  # GPU device
        max_epochs=args.max_epoch,  # Stop training once this number of epochs is reached
        check_val_every_n_epoch=1,  # Check validation every n train epochs.
    )

    # Fit the model
    trainer.fit(model, train_dataloader=train_dl, val_dataloaders=[val_dl])


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
