import os
import sys
import time
import torch
import wandb
import argparse
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

from models import SimpleCNN, SimpleResNet
from dataset import split_dataset, get_dataloaders_for_training


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_epochs = max_epochs
        self.min_lr = min_lr  # avoid zero lr
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            max(
                base_lr * (1 - self.last_epoch / self.max_epochs) ** self.power,
                self.min_lr,
            )
            for base_lr in self.base_lrs
        ]


def train(model, optimizer, criterion, train_loader, device):
    """
    ---------
    Arguments
    ---------
    model: object
        an object of type torch model
    optimizer: object
        an object of type torch Optimizer
    criterion: object
        an object of type torch criterion function
    train_loader: object
        an object of type torch dataloader
    device: object
        an object of type torch device

    -------
    Returns
    -------
    (train_loss, train_acc) : tuple
        a tuple of training loss and training accuracy
    """
    model.to(device)
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    num_train_samples = len(train_loader.dataset)
    num_train_batches = len(train_loader)

    for data, label in train_loader:
        data = data.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, label)
        train_running_loss += loss.item()
        pred_label = torch.argmax(logits, dim=1)
        train_running_correct += (pred_label == label).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / num_train_batches
    train_acc = 100.0 * train_running_correct / num_train_samples
    return train_loss, train_acc


def validate(model, criterion, validation_loader, device):
    """
    ---------
    Arguments
    ---------
    model: object
        an object of type torch model
    criterion: object
        an object of type torch criterion function
    validation_loader: object
        an object of type torch dataloader
    device: object
        an object of type torch device

    -------
    Returns
    -------
    (validation_loss, validation_acc) : tuple
        a tuple of validation loss and validation accuracy
    """
    model.to(device)
    model.eval()
    validation_running_loss = 0.0
    validation_running_correct = 0
    num_validation_samples = len(validation_loader.dataset)
    num_validation_batches = len(validation_loader)

    with torch.no_grad():
        for data, label in validation_loader:
            data = data.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)

            logits = model(data)
            loss = criterion(logits, label)

            validation_running_loss += loss.item()
            pred_label = torch.argmax(logits, dim=1)
            # print(logits, pred_label, label)
            validation_running_correct += (pred_label == label).sum().item()

        validation_loss = validation_running_loss / num_validation_batches
        validation_acc = 100.0 * validation_running_correct / num_validation_samples
    return validation_loss, validation_acc


def train_classifier(ARGS):
    wandb.login()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA device not found, so exiting....")
        sys.exit(0)

    print("Training a CNN model for the Overhead MNIST dataset")
    train_x = []
    train_y = []
    dir_train = os.path.join(ARGS.dir_dataset, "train")
    list_sub_dirs = sorted(os.listdir(dir_train))
    num_classes = len(list_sub_dirs)

    for sub_dir_idx in range(num_classes):
        temp_train_x = os.listdir(os.path.join(dir_train, list_sub_dirs[sub_dir_idx]))
        temp_train_x = [
            os.path.join(list_sub_dirs[sub_dir_idx], f) for f in temp_train_x
        ]
        temp_train_y = [sub_dir_idx] * len(temp_train_x)
        train_x = train_x + temp_train_x
        train_y = train_y + temp_train_y

    (
        train_x,
        validation_x,
        train_y,
        validation_y,
    ) = split_dataset(train_x, train_y)
    num_train_samples = len(train_x)
    num_validation_samples = len(validation_x)
    train_loader, validation_loader = get_dataloaders_for_training(
        train_x,
        train_y,
        validation_x,
        validation_y,
        dir_images=dir_train,
        batch_size=ARGS.batch_size,
    )

    dir_model = os.path.join(ARGS.dir_model, ARGS.model_type)
    if not os.path.isdir(dir_model):
        print(f"Creating directory: {dir_model}")
        os.makedirs(dir_model)

    print(
        f"Num train samples: {num_train_samples}, num validation samples: {num_validation_samples}"
    )
    print(f"Num classes: {num_classes}, model_type: {ARGS.model_type}")

    if ARGS.model_type == "simple_cnn":
        model = SimpleCNN(num_classes=num_classes)
    elif ARGS.model_type == "simple_resnet":
        model = SimpleResNet(num_classes=num_classes)
    elif ARGS.model_type == "medium_resnet":
        model = SimpleResNet(
            list_num_res_units_per_block=[4, 4], num_classes=num_classes
        )
    elif ARGS.model_type == "deep_resnet":
        model = SimpleResNet(
            list_num_res_units_per_block=[6, 6], num_classes=num_classes
        )
    else:
        print(f"Unidentified option for arg (model_type): {ARGS.model_type}")
    model.to(device)

    if ARGS.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=ARGS.learning_rate,
            weight_decay=ARGS.weight_decay,
            momentum=0.9,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay
        )

    if ARGS.lr_scheduler_type == "poly":
        lr_scheduler = PolynomialLR(
            optimizer,
            ARGS.num_epochs + 1,
            power=0.75,
        )

    criterion = torch.nn.CrossEntropyLoss()

    config = {
        "dataset": "Overhead-MNIST",
        "architecture": "CNN",
        "optimizer": ARGS.optimizer_type,
        "lr_scheduler": ARGS.lr_scheduler_type,
        "learning_rate": ARGS.learning_rate,
        "num_epochs": ARGS.num_epochs,
        "batch_size": ARGS.batch_size,
        "weight_decay": ARGS.weight_decay,
    }
    best_validation_acc = 0

    print(
        f"Training the Overhead MNIST image classification model started, model_type: {ARGS.model_type}"
    )
    with wandb.init(project="overhead-mnist-model", config=config):
        for epoch in range(1, ARGS.num_epochs + 1):
            time_start = time.time()
            train_loss, train_acc = train(
                model, optimizer, criterion, train_loader, device
            )
            validation_loss, validation_acc = validate(
                model, criterion, validation_loader, device
            )
            time_end = time.time()
            print(
                f"Epoch: {epoch}/{ARGS.num_epochs}, time: {time_end-time_start:.4f} sec."
            )
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(
                f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_acc:.4f}\n"
            )
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "validation_loss": validation_loss,
                    "validation_acc": validation_acc,
                },
                step=epoch,
            )

            if validation_acc >= best_validation_acc:
                best_validation_acc = validation_acc
                torch.save(
                    model.state_dict(), os.path.join(dir_model, f"{ARGS.model_type}.pt")
                )
                wandb.save(os.path.join(dir_model, f"{ARGS.model_type}.pt"))
            if ARGS.lr_scheduler_type == "poly":
                lr_scheduler.step()
    print("Training the Overhead MNIST image classification model complete!!!!")
    return


def main():
    learning_rate = 1e-3
    weight_decay = 5e-6
    batch_size = 64
    num_epochs = 100
    model_type = "simple_cnn"
    dir_dataset = "/home/abhishek/Desktop/datasets/overhead_mnist/version2"
    dir_model = "trained_models"
    lr_scheduler_type = "none"
    optimizer_type = "adam"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--learning_rate",
        default=learning_rate,
        type=float,
        help="learning rate to use for training",
    )
    parser.add_argument(
        "--weight_decay",
        default=weight_decay,
        type=float,
        help="weight decay to use for training",
    )
    parser.add_argument(
        "--batch_size",
        default=batch_size,
        type=int,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--num_epochs",
        default=num_epochs,
        type=int,
        help="num epochs to train the model",
    )
    parser.add_argument(
        "--dir_dataset",
        default=dir_dataset,
        type=str,
        help="full directory path to dataset containing images",
    )
    parser.add_argument(
        "--dir_model",
        default=dir_model,
        type=str,
        help="full directory path where model needs to be saved",
    )
    parser.add_argument(
        "--model_type",
        default=model_type,
        type=str,
        choices=["simple_cnn", "simple_resnet", "medium_resnet", "deep_resnet"],
        help="model type to be trained",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default=lr_scheduler_type,
        type=str,
        choices=["none", "poly"],
        help="learning rate scheduler to be used for training",
    )
    parser.add_argument(
        "--optimizer_type",
        default=optimizer_type,
        type=str,
        choices=["adam", "sgd"],
        help="optimizer to be used for training",
    )

    ARGS, unparsed = parser.parse_known_args()
    train_classifier(ARGS)
    return


if __name__ == "__main__":
    main()
