import argparse
import os
import pathlib

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

import models_vit
from annotations import combine_intervals

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

ANNOTATIONS = r""

X_OFFSET, Y_OFFSET = (496, 0)

CLASS_NAMES = {2: ["Atrophy", "Normal"], 3: ["iORA + cORA", "iRORA + cRORA", "Normal"],
               5: ["cORA", "cRORA", "iORA", "iRORA", "Normal"]}
CLASS_COLORS = {"iORA": "green", "cORA": "blue", "iRORA": "yellow", "cRORA": "red", "unknown": "pink",
                "multiple": "brown"}


def prepare_ft_model(model_name, num_classes, input_size, chkpt_dir):
    model = models_vit.__dict__[model_name](
        num_classes=num_classes,
        img_size=input_size,
    )

    checkpoint = torch.load(chkpt_dir, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    print("Resume checkpoint %s" % chkpt_dir)

    model.eval()
    return model


def prepare_image(img_path, input_size):
    # load an image
    img = Image.open(img_path)
    img = img.resize((input_size, input_size))
    img = np.array(img) / 255.

    assert img.shape == (input_size, input_size, 3)

    # normalize by mean and sd
    img = img - imagenet_mean
    img = img / imagenet_std

    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x).float()

    return x


def evaluate_class(x, model, img_path, num_classes):
    with torch.no_grad():
        output = model(x)

    output = nn.Softmax(dim=1)(output)
    output = output.squeeze(0).cpu().detach().numpy()

    print(f"Results for {img_path}: {output} -> {CLASS_NAMES[num_classes][np.argmax(output)]}")


def evaluate_IC(x, model, img_path, num_classes):
    with torch.no_grad():
        interval_pred, class_pred = model(x)

    print(f"Interval results for {img_path}: {interval_pred}")
    print(f"Class results for {img_path}: {class_pred} -> {CLASS_NAMES[num_classes][np.argmax(class_pred)]}")


def annotate_images(image_paths, annotations):
    for filename in image_paths:
        basename = os.path.basename(filename)
        bboxes = annotations.findall(f"image[@name='{basename}']/box")

        intervals = []

        # get intervals from bboxes
        for bbox in bboxes:
            # get start and end point
            x0, x1 = float(bbox.get("xtl")), float(bbox.get("xbr"))
            intervals.append([x0, x1])

        intervals = combine_intervals(intervals)
        draw_intervals(filename=filename, intervals=intervals, colors="red", tag="annotated")


def multi_annotate_images(image_paths, annotations):
    for filename in image_paths:
        basename = os.path.basename(filename)
        bboxes = annotations.findall(f"image[@name='{basename}']/box")

        intervals = []
        colors = []

        for bbox in bboxes:
            # get start and end point
            x0, x1 = float(bbox.get("xtl")), float(bbox.get("xbr"))

            # get color
            cls = [attribute.get("name") for attribute in bbox.findall("attribute") if attribute.text == "true"]

            if len(cls) == 0:
                print(f"Bounding box of {filename} has no class assigned")
                cls = ["unknown"]

            if len(cls) > 1:
                print(f"Bounding box of {filename} has multiple classes")
                cls = ["multiple"]

            intervals.append([x0, x1])
            colors.append(CLASS_COLORS[cls[0]])

        draw_intervals(filename=filename, intervals=intervals, colors=colors, tag="multi_annotated")


def draw_intervals(filename, intervals, colors, tag):
    path = os.path.dirname(filename)

    image = Image.open(filename)
    draw = ImageDraw.Draw(image)

    for i, (x0, x1) in enumerate(intervals):
        # draw bbox
        draw.rectangle(xy=((max(x0 - X_OFFSET, 0), 0), (min(x1 - X_OFFSET, image.width - 1), image.height - 1)),
                       outline=colors if type(colors) is str else colors[i])

        # save annotated image
        name, extension = os.path.splitext(filename)
        image.save(os.path.join(path, f"{name}_{tag}{extension}"))


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--model", type=str, default="vit_large_patch16", choices=["vit_large_patch16", "IC_detector"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--resume", type=str)

    args = parser.parse_args()

    # chose fine-tuned model from 'checkpoint'
    model = prepare_ft_model(model_name=args.model, num_classes=args.num_classes, input_size=args.input_size,
                             chkpt_dir=args.resume)

    # get all images
    # supports single files or folders
    data_path = args.data_path
    image_paths = [data_path] if os.path.isfile(data_path) else \
        [str(path) for path in pathlib.Path(data_path).rglob(f"*.*")]

    eval_fn = evaluate_IC if model == "IC_detector" else evaluate_class()

    # run classification for each image
    for img_path in image_paths:
        x = prepare_image(img_path=img_path, input_size=args.input_size)
        eval_fn(x=x, model=model, img_path=img_path, num_classes=args.num_classes)
