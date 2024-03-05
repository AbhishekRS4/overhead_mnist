import os
import json
import logging
import argparse


def gen_label_mapping_json(ARGS: argparse.Namespace) -> None:
    file_json = "label_mapping.json"
    list_labels = sorted(os.listdir(ARGS.dir_train))
    num_labels = len(list_labels)
    dict_label_mapping = {}
    for lbl_idx in range(num_labels):
        dict_label_mapping[lbl_idx] = list_labels[lbl_idx]
    with open(file_json, "w", encoding="utf-8") as file_des:
        json.dump(dict_label_mapping, file_des, ensure_ascii=False, indent=4)
    logging.info(f"Label mapping is saved to {file_json}")
    return


def main() -> None:
    dir_train = "/home/abhishek/Desktop/datasets/overhead_mnist/version2/train"
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dir_train",
        default=dir_train,
        type=str,
        help="full directory path to dataset containing training images",
    )
    ARGS, unparsed = parser.parse_known_args()
    gen_label_mapping_json(ARGS)
    return


if __name__ == "__main__":
    main()
