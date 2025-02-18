import os

from PIL import Image

ORG_PATH = r""
NEW_PATH = r""

ORG_X, ORG_Y = (496, 0)
ORG_WIDTH, ORG_HEIGHT = (512, 512)

NEW_WIDTH, NEW_HEIGHT = (512, 512)
NEW_FORMAT = "png"


def load_images():
    result = []

    for path, subdirs, files in os.walk(ORG_PATH):
        for name in files:
            result.append((path, name, Image.open(os.path.join(path, name))))

    return result


def crop_images(images):
    result = []

    for path, name, image in images:
        result.append((path, name, image.crop((ORG_X, ORG_Y, ORG_X + ORG_WIDTH, ORG_Y + ORG_HEIGHT))))

    return result


def resize_images(images):
    result = []

    for path, name, image in images:
        result.append((path, name, image.resize((NEW_WIDTH, NEW_HEIGHT), Image.Resampling.LANCZOS)))

    return result


def save_images(images):
    for path, name, image in images:
        new_path = path.replace(ORG_PATH, NEW_PATH)

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        new_name = f"{os.path.splitext(name)[0]}.{NEW_FORMAT}"
        image.save(os.path.join(new_path, new_name))


if __name__ == "__main__":
    org_images = load_images()
    # cropped_images = crop_images(org_images)
    resized_images = resize_images(org_images)
    save_images(resized_images)
