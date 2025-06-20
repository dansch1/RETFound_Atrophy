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


def evaluate(x, model, image, args, annotations=None):
    with torch.no_grad():
        output = model(x)

    output_ = nn.Softmax(dim=1)(output)
    output_label = output_.argmax(dim=1)

    pred_label = output_label.squeeze(0).detach().cpu().numpy()

    image_path = pathlib.Path(image)
    print(f"Class results for {image_path}: {output_} -> {pred_label} ({'Atrophy' if pred_label == 0 else 'Normal'})")

    if not annotations:
        return

    true_label = 0 if image_path.name in annotations else 1
    print(f"Correct class is: {true_label} ({'Atrophy' if true_label == 0 else 'Normal'})")


def evaluate_I(x, model, image, args, annotations=None):
    with torch.no_grad():
        output = model(x)

    output_ = output.reshape(-1, 2)
    output_intervals = output_[(output_[:, 0] >= 0) & (output_[:, 1] >= 0)]  # remove dummy intervals

    pred_intervals = output_intervals.cpu().detach().tolist()
    prediction = [interval + [0] for interval in pred_intervals]  # add atrophy class

    image_path = pathlib.Path(image)
    print(f"Prediction for {image_path}: {output_} -> {prediction}")

    num_classes = args.nb_classes
    output_dir = args.output_dir

    if args.draw:
        draw_results(image_path=image_path, results=prediction, num_classes=num_classes,
                     output_dir=output_dir, tag="prediction")

    if not annotations:
        return

    target = annotations.get(image_path.name, [])
    print(f"Correct are: {target}")

    if args.draw:
        draw_results(image_path=image_path, results=target, num_classes=num_classes,
                     output_dir=output_dir, tag="target")


def evaluate_IC(x, model, image, args, annotations=None):
    num_classes = args.nb_classes

    with torch.no_grad():
        interval_pred, class_pred = model(x)
        class_pred = class_pred.reshape(-1, num_classes)

    class_pred_ = nn.Softmax(dim=1)(class_pred)
    output_label = class_pred_.argmax(dim=1)
    interval_pred_ = interval_pred.reshape(-1, 2)

    output_ = torch.cat(tensors=(interval_pred_, output_label.unsqueeze(1)), dim=1)
    prediction_ = output_[(output_[:, 0] >= 0) & (output_[:, 1] >= 0) & (output_[:, 2] != 0)]  # remove dummy intervals
    prediction = prediction_.cpu().detach().tolist()

    image_path = pathlib.Path(image)
    print(f"Prediction for {image_path}: {output_} -> {prediction}")

    output_dir = args.output_dir

    if args.draw:
        draw_results(image_path=image_path, results=prediction, num_classes=num_classes,
                     output_dir=output_dir, tag="prediction")

    if not annotations:
        return

    target = annotations.get(image_path.name, [])
    print(f"Correct are: {target}")

    if args.draw:
        draw_results(image_path=image_path, results=target, num_classes=num_classes,
                     output_dir=output_dir, tag="target")


def draw_results(image_path, results, num_classes, output_dir, tag):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # create output directory if necessary
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, (x0, x1, cls) in enumerate(results):
        if cls == 0:
            continue

        lower, upper = min(x0, x1), max(x0, x1)

        # draw bbox
        draw.rectangle(xy=((max(lower, 0), 0), (min(upper, image.width - 1), image.height - 1)),
                       outline=CLASS_COLORS[num_classes][cls], width=4)

    # save annotated image
    image.save(os.path.join(output_dir, f"{image_path.stem}_{tag}{image_path.suffix}"))


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
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--draw", action='store_true')
    parser.add_argument('--output_dir', default='./results',
                        help='path where to save results')

    args = parser.parse_args()

    # chose fine-tuned model from checkpoint
    device = torch.device(args.device)
    model = prepare_model(args)
    model.to(device)

    with open(args.annotations) as f:
        annotations = json.loads(f.read())

    # get all images
    # supports single files or folders
    images = get_all_files(args.data_path)
    transform = build_transform("eval", args)

    if args.model == "I_detector":
        eval_fn = evaluate_I
    elif args.model == "IC_detector":
        eval_fn = evaluate_IC
    else:
        eval_fn = evaluate

    # run classification for each image
    for image in images:
        # prepare image
        x = transform(Image.open(image).convert("RGB")).unsqueeze(0)
        x = x.to(device, non_blocking=True)

        # evaluate model with image
        eval_fn(x=x, model=model, image=image, args=args, annotations=annotations)
