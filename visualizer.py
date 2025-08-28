import argparse
import json
import os
import pathlib

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.interpolate import UnivariateSpline
from torch import nn

import models_vit
from preprocess_data import setup_output_dir

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
        draw_results(image_path=image_path, input_size=args.input_size, results=prediction, num_classes=num_classes,
                     output_dir=output_dir, tag="prediction", segment_layers=args.segment_layers)

    if not annotations:
        return

    target = annotations.get(image_path.name, [])
    print(f"Correct are: {target}")

    if args.draw:
        draw_results(image_path=image_path, input_size=args.input_size, results=target, num_classes=num_classes,
                     output_dir=output_dir, tag="target", segment_layers=False)


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
        draw_results(image_path=image_path, input_size=args.input_size, results=prediction, num_classes=num_classes,
                     output_dir=output_dir, tag="prediction", segment_layers=args.segment_layers)

    if not annotations:
        return

    target = annotations.get(image_path.name, [])
    print(f"Correct are: {target}")

    if args.draw:
        draw_results(image_path=image_path, input_size=args.input_size, results=target, num_classes=num_classes,
                     output_dir=output_dir, tag="target", segment_layers=False)


def draw_results(image_path, input_size, results, num_classes, output_dir, tag, segment_layers, line_width=4):
    image = Image.open(image_path)
    scale_factor = image.width / input_size

    # create output directory if necessary
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # get segmentation
    upper_line, lower_line = segment_oct_layers(str(image_path)) if segment_layers else ([], [])

    # draw result
    draw = ImageDraw.Draw(image)

    for i, (x0, x1, cls) in enumerate(results):
        if cls == 0:
            continue

        x0, x1 = sorted((x0, x1))

        x0 *= scale_factor
        x1 *= scale_factor

        x0, x1 = max(int(x0), 0), min(int(x1), image.width - 1)

        color = CLASS_COLORS[num_classes][cls]

        # draw vertical lines
        for x in (x0, x1):
            y_upper = upper_line[x] if x < len(upper_line) and not np.isnan(upper_line[x]) else image.height - 1
            y_lower = lower_line[x] if x < len(lower_line) and not np.isnan(lower_line[x]) else 0
            draw.line([(x, y_upper), (x, y_lower)], fill=color, width=line_width)

        # draw horizontal lines
        upper_points = [(x, upper_line[x]) for x in range(x0, x1 + 1) if
                        x < len(upper_line) and not np.isnan(upper_line[x])]
        lower_points = [(x, lower_line[x]) for x in range(x0, x1 + 1) if
                        x < len(lower_line) and not np.isnan(lower_line[x])]

        if len(upper_points) < 2:
            upper_points = [(x0, 0), (x1, 0)]
        if len(lower_points) < 2:
            lower_points = [(x0, image.height - 1), (x1, image.height - 1)]

        for points in (upper_points, lower_points):
            draw.line(points, fill=color, width=line_width)

    # save annotated image
    image.save(os.path.join(output_dir, f"{image_path.stem}_{tag}{image_path.suffix}"))


def segment_oct_layers(image_path):
    """
    Segments the uppermost and lowermost retinal layers from an OCT B-scan.

    Parameters:
        image_path (str): Path to the OCT image file.

    Returns:
        upper_line (np.ndarray): Y-coordinates of the upper retinal layer.
        lower_line (np.ndarray): Y-coordinates of the lower retinal layer.
    """

    # step 0: load image
    oct_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # step 1: preprocessing
    denoised = cv2.GaussianBlur(oct_image, (5, 5), 0)
    brightened = cv2.add(denoised, 70)
    filled = cv2.morphologyEx(brightened, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    smoothed = cv2.GaussianBlur(filled, (9, 9), 3)
    _, binary = cv2.threshold(smoothed, 140, 255, cv2.THRESH_BINARY)

    # step 2: morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)
    dilated = cv2.dilate(opened, np.ones((1, 2), np.uint8), iterations=2)

    # step 3: largest connected component
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    mask = np.zeros_like(dilated)
    cv2.drawContours(mask, largest, -1, 255, thickness=cv2.FILLED)

    # step 4: edge detection
    edges = cv2.Canny(mask, 100, 200)

    # step 5: find upper and lower edges
    upper_y = []
    lower_y = []

    for col in range(edges.shape[1]):
        rows = np.where(edges[:, col] > 0)[0]

        if len(rows) > 0:
            upper_y.append(min(rows))
            lower_y.append(max(rows))
        else:
            upper_y.append(np.nan)
            lower_y.append(np.nan)

    # step 6: smoothing
    x = np.arange(len(upper_y))
    upper_y = np.array(upper_y, dtype=np.float64)
    lower_y = np.array(lower_y, dtype=np.float64)

    valid_upper = ~np.isnan(upper_y)
    valid_lower = ~np.isnan(lower_y)

    start_u = np.argmax(valid_upper)
    end_u = len(valid_upper) - np.argmax(valid_upper[::-1]) - 1
    start_l = np.argmax(valid_lower)
    end_l = len(valid_lower) - np.argmax(valid_lower[::-1]) - 1

    x_u = x[start_u:end_u + 1]
    y_u = upper_y[start_u:end_u + 1]
    x_l = x[start_l:end_l + 1]
    y_l = lower_y[start_l:end_l + 1]

    upper_line = np.full_like(upper_y, np.nan)
    lower_line = np.full_like(lower_y, np.nan)

    if np.sum(~np.isnan(y_u)) > 5:
        s_upper = UnivariateSpline(x_u[~np.isnan(y_u)], y_u[~np.isnan(y_u)], s=300)
        upper_line[start_u:end_u + 1] = s_upper(x_u)

    if np.sum(~np.isnan(y_l)) > 5:
        s_lower = UnivariateSpline(x_l[~np.isnan(y_l)], y_l[~np.isnan(y_l)], s=300)
        lower_line[start_l:end_l + 1] = s_lower(x_l)

    return upper_line, lower_line


def get_all_files(data_path):
    path = pathlib.Path(data_path)

    if path.is_file():
        return [str(path.resolve())]

    if path.is_dir():
        return [str(p.resolve()) for p in path.rglob("*") if p.is_file()]

    return []


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument('--device', default='cuda',
                        help='device to use for testing')
    parser.add_argument("--model", type=str, default="RETFound_mae",
                        choices=["RETFound_mae", "I_detector", "IC_detector"])
    parser.add_argument("--nb_classes", type=int, default=2)
    parser.add_argument("--max_intervals", type=int, default=10)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--draw", action='store_true')
    parser.add_argument("--segment_layers", action='store_true')
    parser.add_argument('--output_dir', default='./results',
                        help='path where to save results')

    args = parser.parse_args()

    # chose fine-tuned model from checkpoint
    device = torch.device(args.device)
    model = prepare_model(args)
    model.to(device)

    annotations = None
    if args.annotations:
        with open(args.annotations) as f:
            annotations = json.loads(f.read())

    if args.draw:
        setup_output_dir(args.output_dir)

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
