import argparse
import json
import os
import pathlib

import torch
from PIL import Image, ImageDraw
from torch import nn

import models_vit

from util.datasets import build_transform

CLASS_NAMES = {2: ["Normal", "Atrophy"], 3: ["Normal", "iORA + cORA", "iRORA + cRORA"],
               5: ["Normal", "cORA", "cRORA", "iORA", "iRORA"]}

CLASS_COLORS = {2: {0: "white", 1: "red"},  # 1 class: (0.Normal), 1.atrophy
                3: {0: "white", 1: "green", 2: "red"},  # 2 classes: 1.iORA+cORA, 2.iRORA+cRORA
                5: {0: "white", 1: "green", 2: "blue", 3: "yellow",
                    4: "red"}}  # 4 classes: 1.iORA, 2.cORA, 3.iRORA, 4.cRORA


def prepare_model(args):
    model = models_vit.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        args=args,
    )

    chkpt_dir = args.resume
    checkpoint = torch.load(chkpt_dir, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    print("Resume checkpoint %s" % chkpt_dir)

    model.eval()
    return model


def evaluate(x, model, image_path, annotations, num_classes):
    with torch.no_grad():
        output = model(x)

    output_ = nn.Softmax(dim=1)(output)
    output_label = output_.argmax(dim=1)

    pred_label = output_label.squeeze(0).detach().cpu().numpy()

    image_name = pathlib.Path(image_path).name
    true_label = max(interval[2] for interval in annotations[image_name]) if image_name in annotations else 0

    print(f"Class results for {image_path}: {output} -> {CLASS_NAMES[num_classes][pred_label]}")
    print(f"Correct class is: {CLASS_NAMES[num_classes][true_label]}")


def evaluate_I(x, model, image_path, annotations, num_classes):
    with torch.no_grad():
        output = model(x)

    pred_interval = output.reshape(-1, 2).cpu().detach().numpy()

    image_name = pathlib.Path(image_path).name
    true_interval = annotations.get(image_name, [])[:2]

    print(f"Interval results for {image_path}: {output}")
    print(f"Correct interval is: {true_interval}")

    # TODO: draw results


def evaluate_IC(x, model, image_path, annotations, num_classes):
    with torch.no_grad():
        interval_pred, class_pred = model(x)
        class_pred = class_pred.reshape(-1, num_classes)

    class_pred_ = nn.Softmax(dim=1)(class_pred)
    output_label = class_pred_.argmax(dim=1)

    pred_label = output_label.squeeze(0).detach().cpu().numpy()
    pred_interval = interval_pred.reshape(-1, 2).cpu().detach().numpy()

    image_name = pathlib.Path(image_path).name
    true_values = annotations.get(image_name, [])

    print(f"Class results for {image_path}: {class_pred} -> {CLASS_NAMES[num_classes][pred_label]}")
    print(f"Interval results for {image_path}: {interval_pred}")
    print(f"Correct is: {true_values}")

    # TODO: draw results


def draw_results(image_path, results, num_classes, tag):
    path = os.path.dirname(image_path)

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for i, (x0, x1, cls) in enumerate(results):
        # draw bbox
        draw.rectangle(xy=((max(x0, 0), 0), (min(x1, image.width - 1), image.height - 1)),
                       outline=CLASS_COLORS[num_classes][cls], width=4)

        # save annotated image
        name, extension = os.path.splitext(image_path)
        image.save(os.path.join(path, f"{name}_{tag}{extension}"))


def get_all_files(data_path):
    path = pathlib.Path(data_path)
    if path.is_file():
        return [str(path.resolve())]
    elif path.is_dir():
        return [str(p.resolve()) for p in path.rglob("*") if p.is_file()]
    else:
        return []


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument('--device', default='cuda',
                        help='device to use for testing')
    parser.add_argument("--model", type=str, default="RETFound_mae",
                        choices=["RETFound_mae", "I_detector", "IC_detector"])
    parser.add_argument("--nb_classes", type=int, default=2)
    parser.add_argument("--max_intervals", type=int, default=10)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--annotations", type=str)

    args = parser.parse_args()

    # chose fine-tuned model from checkpoint
    device = torch.device(args.device)
    model = prepare_model(args)
    model.to(device)

    with open(args.annotations) as f:
        annotations = json.loads(f.read())

    # get all images
    # supports single files or folders
    image_paths = get_all_files(args.data_path)
    transform = build_transform("eval", args)

    if args.model == "I_detector":
        eval_fn = evaluate_I
    elif args.model == "IC_detector":
        eval_fn = evaluate_IC
    else:
        eval_fn = evaluate

    # run classification for each image
    for image_path in image_paths:
        # prepare image
        x = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
        x = x.to(device, non_blocking=True)

        # evaluate model with image
        eval_fn(x=x, model=model, image_path=image_path, annotations=annotations, num_classes=args.nb_classes)
