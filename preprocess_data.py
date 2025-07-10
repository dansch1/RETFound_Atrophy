import json
import os
import random
import shutil
import warnings

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import argparse

from torchvision.transforms import transforms
import torchvision.transforms.functional as TF


def load_images(org_path):
    result = []

    for filename in os.listdir(org_path):
        path = os.path.join(org_path, filename)
        result.append((Image.open(path), filename))

    return result


def load_annotations(annotations_path, decimal_places=3):
    with open(args.annotations) as f:
        annotations = json.loads(f.read())

    # round intervals
    return {filename: [[round(x0, decimal_places), round(x1, decimal_places), c] for [x0, x1, c] in intervals] for
            filename, intervals in annotations.items()}


def crop_images(images, x_offset, y_offset, org_size):
    result = []

    for image, filename in images:
        result.append((image.crop((x_offset, y_offset, x_offset + org_size, y_offset + org_size)), filename))

    return result


def scale_images(images, org_size, new_size, annotations):
    result = []

    transform = transforms.Compose([
        transforms.Lambda(lambda img: ImageOps.pad(img, (new_size, new_size), color=(0, 0, 0))),
        transforms.ToTensor()
    ])

    # scale images
    for image, filename in images:
        result.append((TF.to_pil_image(transform(image)), filename))

    scale_factor = float(new_size) / org_size
    new_annotations = {filename: [[x0 * scale_factor, x1 * scale_factor, c] for [x0, x1, c] in intervals] for
                       filename, intervals in annotations.items()}

    return result, new_annotations


def split_images(images, train_ratio, val_ratio, test_ratio):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    return splits


def apply_flip(image, intervals, width, decimal_places=3):
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_annots = [[round(width - x1, decimal_places), round(width - x0, decimal_places), cls] for [x0, x1, cls] in
                      intervals] if intervals else None

    return flipped_image, flipped_annots, "_flip"


def apply_noise(image, intervals, **kwargs):
    noisy = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 5, noisy.shape)
    noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy)

    return noisy_image, intervals, "_noise"


def apply_blur(img, intervals, **kwargs):
    blurred = img.filter(ImageFilter.GaussianBlur(radius=1))

    return blurred, intervals, "_blur"


def augment_images(images, new_size, annotations, aug_config=None):
    result = images.copy()
    new_annotations = annotations.copy()

    if aug_config is None:
        aug_config = {
            apply_flip: 0.5,
            apply_noise: 0.3,
            # apply_blur: 0.2,
        }

    for image, filename in images:
        intervals = annotations.get(filename, None)
        image_variants = [(image, intervals, "")]

        for aug_func, prob in aug_config.items():
            next_variants = []

            for image_var, intervals_var, suffix_var in image_variants:
                if random.random() < prob:
                    aug_image, aug_intervals, aug_suffix = aug_func(image_var, intervals_var, width=new_size)
                    new_suffix = suffix_var + aug_suffix
                    next_variants.append((aug_image, aug_intervals, new_suffix))

            image_variants.extend(next_variants)

        base_name, extension = os.path.splitext(filename)

        for aug_image, aug_intervals, aug_suffix in image_variants[1:]:  # skip original
            new_filename = f"{base_name}{aug_suffix}{extension}"
            result.append((aug_image, new_filename))

            if aug_intervals:
                new_annotations[new_filename] = aug_intervals

    return result, new_annotations


def save_images(splits, new_path, annotations, format):
    output_dir = os.path.join(new_path, "splits")
    setup_output_dir(output_dir)

    for split_name, split_images in splits.items():
        split_path = os.path.join(output_dir, split_name)

        if not os.path.exists(split_path):
            os.makedirs(split_path)

        for image, filename in split_images:
            intervals = annotations.get(filename, [])
            max_cls = "0" if not intervals or len(intervals) == 0 else str(max(interval[2] for interval in intervals))

            cls_path = os.path.join(split_path, max_cls)

            if not os.path.exists(cls_path):
                os.makedirs(cls_path)

            new_filename = f"{os.path.splitext(filename)[0]}.{format}"
            image.save(os.path.join(cls_path, new_filename))


def setup_output_dir(output_dir):
    # create output dir if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return

    # clear all files in the output dir
    for f in os.listdir(output_dir):
        file_path = os.path.join(output_dir, f)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            warnings.warn(f"Failed to delete {file_path}. Reason: {e}")


def save_annotations(annotations, filename="annotations.json"):
    with open(os.path.join(args.new_path, filename), "w") as f:
        json.dump(annotations, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--org_path", type=str)
    parser.add_argument("--new_path", type=str)
    parser.add_argument("--annotations", type=str)
    parser.add_argument("--format", type=str, default="png")
    parser.add_argument("--org_size", type=int, default=512)
    parser.add_argument("--new_size", type=int, default=512)
    parser.add_argument("--x_offset", type=int, default=496)
    parser.add_argument("--y_offset", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    # loading
    org_images = load_images(args.org_path)
    annotations = load_annotations(args.annotations)

    # preprocessing
    images = crop_images(images=org_images, x_offset=args.x_offset, y_offset=args.y_offset, org_size=args.org_size)

    images, annotations = scale_images(images=images, org_size=args.org_size, new_size=args.new_size,
                                       annotations=annotations)

    splits = split_images(images=images, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
                          test_ratio=args.test_ratio)

    # splits["train"], annotations = augment_images(images=splits["train"], new_size=args.new_size, annotations=annotations)

    # saving
    save_images(splits=splits, new_path=args.new_path, annotations=annotations, format=args.format)
    save_annotations(annotations)
