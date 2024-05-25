import os
import sys
import time
import torch
import logging
import argparse

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from train import validate
from dataset import get_dataloader_for_testing
from models import SimpleCNN, SimpleResNet, ComplexResNet, ComplexResNetV2, SimpleResKANet, ComplexResKANet, ComplexResKANetV2


def test_classifier(ARGS: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_x = []
    test_y = []
    list_sub_dirs = sorted(os.listdir(ARGS.dir_test_set))
    num_classes = len(list_sub_dirs)

    for sub_dir_idx in range(num_classes):
        temp_test_x = os.listdir(
            os.path.join(ARGS.dir_test_set, list_sub_dirs[sub_dir_idx])
        )
        temp_test_x = [os.path.join(list_sub_dirs[sub_dir_idx], f) for f in temp_test_x]
        temp_test_y = [sub_dir_idx] * len(temp_test_x)
        test_x = test_x + temp_test_x
        test_y = test_y + temp_test_y

    test_loader = get_dataloader_for_testing(
        test_x, test_y, dir_images=ARGS.dir_test_set
    )

    if ARGS.model_type == "simple_cnn":
        model = SimpleCNN(num_classes=num_classes)
    elif ARGS.model_type == "simple_resnet":
        model = SimpleResNet(num_classes=num_classes)
    elif ARGS.model_type == "medium_simple_resnet":
        model = SimpleResNet(
            list_num_res_units_per_block=[4, 4], num_classes=num_classes
        )
    elif ARGS.model_type == "deep_simple_resnet":
        model = SimpleResNet(
            list_num_res_units_per_block=[6, 6], num_classes=num_classes
        )
    elif ARGS.model_type == "complex_resnet":
        model = ComplexResNet(
            list_num_res_units_per_block=[4, 4, 4], num_classes=num_classes
        )
    elif ARGS.model_type == "complex_resnet_v2":
        model = ComplexResNetV2(
            list_num_res_units_per_block=[4, 4, 4], num_classes=num_classes
        )
    elif ARGS.model_type == "simple_reskanet":
        model = SimpleResKANet(num_classes=num_classes)
    elif ARGS.model_type == "medium_simple_reskanet":
        model = SimpleResKANet(
            list_num_res_units_per_block=[4, 4], num_classes=num_classes
        )
    elif ARGS.model_type == "deep_simple_reskanet":
        model = SimpleResKANet(
            list_num_res_units_per_block=[6, 6], num_classes=num_classes
        )
    elif ARGS.model_type == "complex_reskanet":
        model = ComplexResKANet(
            list_num_res_units_per_block=[4, 4, 4], num_classes=num_classes
        )
    elif ARGS.model_type == "complex_reskanet_v2":
        model = ComplexResKANetV2(
            list_num_res_units_per_block=[4, 4, 4], num_classes=num_classes
        )
    else:
        logging.info(f"Unidentified option for arg (model_type): {ARGS.model_type}")
    model.load_state_dict(torch.load(ARGS.file_model, map_location=device))
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    num_test_files = len(test_x)
    logging.info(f"Num test files: {num_test_files}")
    logging.info(
        f"Testing the Overhead MNIST image classification model started, model_type: {ARGS.model_type}"
    )
    _, test_acc = validate(model, criterion, test_loader, device)
    logging.info(f"Test Accuracy: {test_acc:.4f}\n")
    logging.info("Testing the Overhead MNIST image classification model complete!!!!")
    return


def main() -> None:
    model_type = "simple_cnn"
    dir_test_set = "/home/abhishek/Desktop/datasets/overhead_mnist/version2/test/"
    file_model = "simple_cnn.pt"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dir_test_set",
        default=dir_test_set,
        type=str,
        help="full directory path containing test set images",
    )
    parser.add_argument(
        "--file_model",
        default=file_model,
        type=str,
        help="full path to model file for loading the checkpoint",
    )
    parser.add_argument(
        "--model_type",
        default=model_type,
        type=str,
        choices=[
            "simple_cnn",
            "simple_resnet",
            "medium_simple_resnet",
            "deep_simple_resnet",
            "complex_resnet",
            "complex_resnet_v2",
        ],
        help="model type to be tested and evaluated",
    )

    ARGS, unparsed = parser.parse_known_args()
    test_classifier(ARGS)
    return


if __name__ == "__main__":
    main()
