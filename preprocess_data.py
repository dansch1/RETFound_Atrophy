import os

from PIL import Image, ImageOps
import argparse

from torchvision.transforms import transforms
import torchvision.transforms.functional as TF


def load_images(org_path):
    result = []

    for path, subdirs, files in os.walk(org_path):
        for name in files:
            result.append((path, name, Image.open(os.path.join(path, name))))

    return result


def crop_images(images, offset, org_size):
    result = []

    x_offset, y_offset = offset
    width, height = org_size

    for path, name, image in images:
        result.append((path, name, image.crop((x_offset, y_offset, x_offset + width, y_offset + height))))

    return result


def resize_images(images, new_size):
    result = []

    transform = transforms.Compose([
        transforms.Lambda(lambda img: ImageOps.pad(img, new_size, color=(0, 0, 0))),  # schwarzes Padding
        transforms.ToTensor()
    ])

    for path, name, image in images:
        result.append((path, name, TF.to_pil_image(transform(image))))

    return result


def save_images(images, org_path, new_path, format):
    for path, name, image in images:
        save_path = path.replace(org_path, new_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_name = f"{os.path.splitext(name)[0]}.{format}"
        image.save(os.path.join(save_path, save_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--org_path", type=str)
    parser.add_argument("--new_path", type=str)
    parser.add_argument("--format", type=str, default="png")
    parser.add_argument("--org_size", type=tuple, default=(512, 512))
    parser.add_argument("--new_size", type=tuple, default=(512, 512))
    parser.add_argument("--offset", type=tuple, default=(496, 0))
    args = parser.parse_args()

    org_images = load_images(args.org_path)
    cropped_images = crop_images(images=org_images, offset=args.offset, org_size=args.org_size)
    resized_images = resize_images(images=cropped_images, new_size=args.new_size)
    save_images(images=resized_images, org_path=args.org_path, new_path=args.new_path, format=args.format)
