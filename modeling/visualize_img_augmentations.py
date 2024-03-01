import os
import io
import numpy as np
import streamlit as st
import torchvision.transforms as transforms

from PIL import Image
from skimage.io import imread


def visualize_augmentations():
    image_orig = None
    st.title("Image augmentation visualizer")

    augmenter = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation((90, 90)),
            transforms.RandomAffine(
                degrees=(-5, 5),
                translate=(0, 0.05),
                scale=(0.9, 1.05),
                shear=(-5, 5),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=170,
            ),
        ]
    )

    st.header("Upload an input image and apply augmentation")
    # select an input image file
    image_file_buffer = st.sidebar.file_uploader(
        "Select input image", type=["jpg", "jpeg"]
    )

    # read the image and apply augmentation
    if image_file_buffer is not None:
        image_orig = Image.open(image_file_buffer)

    if image_orig is not None:
        image_array_orig = np.array(image_orig)
        st.image(
            image_array_orig,
            caption=f"Original input image: {image_file_buffer.name}",
            width=8 * image_array_orig.shape[0],
        )
        # st.write(image_array_orig.shape)

        augment_button = st.sidebar.button("Apply augmentation")
        if augment_button:
            image_aug = augmenter(image_array_orig)
            image_array_aug = np.squeeze(np.array(image_aug))
            st.image(
                image_array_aug,
                caption=f"Augmented image",
                width=8 * image_array_orig.shape[0],
            )
            # st.write(image_array_aug.shape)
    return


def main():
    visualize_augmentations()
    return


if __name__ == "__main__":
    main()
