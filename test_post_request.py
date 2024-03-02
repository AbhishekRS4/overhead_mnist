import os
import argparse
import requests
import numpy as np


def send_image_post_request(ARGS):
    list_test_images = os.listdir(ARGS.dir_test_images)

    for file_image in list_test_images:
        file_image = os.path.join(ARGS.dir_test_images, file_image)
        files = {"image_file": (file_image, open(file_image, "rb"))}

        response = requests.post(
            "http://127.0.0.1:7860/predict",
            files=files,
        )

        print(response.json())
    return


def main():
    dir_test_images = "./sample_test_images/"
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dir_test_images",
        default=dir_test_images,
        type=str,
        help="full directory path to dataset containing test images",
    )
    ARGS, unparsed = parser.parse_known_args()
    send_image_post_request(ARGS)
    return


if __name__ == "__main__":
    main()
