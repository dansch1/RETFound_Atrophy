import argparse
import os
import pathlib

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

import models_vit
from annotations import get_class_intervals

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

CLASS_NAMES = {2: ["Atrophy", "Normal"], 3: ["iORA + cORA", "iRORA + cRORA", "Normal"],
               5: ["cORA", "cRORA", "iORA", "iRORA", "Normal"]}

CLASS_COLORS = {2: {0: "white", 1: "red"},  # 1 class: (0.Normal), 1.atrophy
                3: {0: "white", 1: "green", 2: "red"},  # 2 classes: 1.iORA+cORA, 2.iRORA+cRORA
                5: {0: "white", 1: "green", 2: "blue", 3: "yellow",
                    4: "red"}}  # 4 classes: 1.iORA, 2.cORA, 3.iRORA, 4.cRORA


def prepare_model(model_name, num_classes, input_size, chkpt_dir, kwargs):
    model = models_vit.__dict__[model_name](
        num_classes=num_classes,
        img_size=input_size,
        **kwargs
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


def evaluate_class(x, model, img_path, num_classes, annotations):
    with torch.no_grad():
        output = model(x)

    output = nn.Softmax(dim=1)(output)
    output = output.squeeze(0).cpu().detach().numpy()

    print(f"Results for {img_path}: {output} -> {CLASS_NAMES[num_classes][np.argmax(output)]}")


def evaluate_IC(x, model, img_path, num_classes, annotations):
    with torch.no_grad():
        interval_pred, class_pred = model(x)

    print(f"Interval results for {img_path}: {interval_pred}")
    print(f"Class results for {img_path}: {class_pred} -> {CLASS_NAMES[num_classes][np.argmax(class_pred)]}")

    target = get_class_intervals(image=img_path, annotations=annotations, num_classes=num_classes)
    draw_results(image_path=img_path, results=target, num_classes=num_classes, tag="target")

    draw_results(image_path=img_path, results=zip(interval_pred, class_pred), num_classes=num_classes, tag="prediction")


def draw_results(image_path, results, num_classes, tag):
    path = os.path.dirname(image_path)

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for i, (x0, x1, cls) in enumerate(results):
        # draw bbox
        draw.rectangle(xy=((max(x0, 0), 0), (min(x1, image.width - 1), image.height - 1)),
                       outline=CLASS_COLORS[num_classes][cls])

        # save annotated image
        name, extension = os.path.splitext(image_path)
        image.save(os.path.join(path, f"{name}_{tag}{extension}"))


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--model", type=str, default="vit_large_patch16", choices=["vit_large_patch16", "IC_detector"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--annotations", type=str)
    parser.add_argument("--max_intervals", type=int, default=10)

    args = parser.parse_args()

    # chose fine-tuned model from 'checkpoint'
    kwargs = {"max_intervals": args.max_intervals} if args.model == "IC_detector" else {}
    model = prepare_model(model_name=args.model, num_classes=args.num_classes, input_size=args.input_size,
                          chkpt_dir=args.resume, **kwargs)

    # get all images
    # supports single files or folders
    data_path = args.data_path
    image_paths = [data_path] if os.path.isfile(data_path) else \
        [str(path) for path in pathlib.Path(data_path).rglob(f"*.*")]

    eval_fn = evaluate_IC if model == "IC_detector" else evaluate_class

    # run classification for each image
    for img_path in image_paths:
        x = prepare_image(img_path=img_path, input_size=args.input_size)
        eval_fn(x=x, model=model, img_path=img_path, num_classes=args.num_classes, annotations=args.annotations)
