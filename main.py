import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32)


def align_image(reference_gray, image):
    image_gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        100,
        1e-6
    )

    try:
        _, warp_matrix = cv2.findTransformECC(
            reference_gray,
            image_gray,
            warp_matrix,
            cv2.MOTION_TRANSLATION,
            criteria
        )

        aligned = cv2.warpAffine(
            image,
            warp_matrix,
            (reference_gray.shape[1], reference_gray.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )

        return aligned

    except cv2.error:
        return image


def stack_images(input_dir, output_file):
    input_dir = Path(input_dir)

    files = sorted(
        list(input_dir.glob("*.tif")) +
        list(input_dir.glob("*.tiff")) +
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg"))
    )

    if not files:
        raise ValueError("No image files found in the selected folder.")

    print(f"Found {len(files)} images")

    reference = load_image(files[0])
    reference_gray = cv2.cvtColor(reference.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    stack = np.zeros_like(reference, dtype=np.float32)
    count = 0

    for file in files:
        print(f"Processing: {file.name}")

        img = load_image(file)

        if img.shape != reference.shape:
            print(f"Skipped {file.name}: image size does not match reference")
            continue

        aligned = align_image(reference_gray, img)
        stack += aligned
        count += 1

    if count == 0:
        raise ValueError("No valid images were stacked.")

    stacked = stack / count
    stacked = np.clip(stacked, 0, 255).astype(np.uint8)

    Image.fromarray(stacked).save(output_file)

    print(f"Done. Stacked {count} images.")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Folder containing TIFF/images")
    parser.add_argument(
        "-o",
        "--output",
        default="stacked_output.tif",
        help="Output file name"
    )

    args = parser.parse_args()
    stack_images(args.input_dir, args.output)
