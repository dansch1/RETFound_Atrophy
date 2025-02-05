import argparse
import os
import pathlib

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn

import models_vit

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

CLASS_NAMES = ["Atrophy", "Normal"]


def prepare_ft_model(chkpt_dir, num_classes):
    model = models_vit.__dict__["vit_large_patch16"](
        num_classes=num_classes,
        drop_path_rate=0.1,
        global_pool=True,
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

    assert img.shape == (224, 224, 3)

    # normalize by mean and sd
    img = img - imagenet_mean
    img = img / imagenet_std

    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x).float()

    return x


def run_classification(img_path, input_size, model):
    x = prepare_image(img_path=img_path, input_size=input_size)

    # model inferenceda
    with torch.no_grad():
        output = model(x)

    output = nn.Softmax(dim=1)(output)
    output = output.squeeze(0).cpu().detach().numpy()

    print(f"Results for {img_path}: {output} -> {CLASS_NAMES[np.argmax(output)]}")

    # visualization
    """
    categories = ["Atrophy", "Normal"]
    colors = ["red", "green"]
    prob_result = draw_result(output, categories, colors)

    Image.fromarray(prob_result).save('classification.png')
    """


def draw_result(probabilities, categories, colors):
    # Creating the bar plot
    fig = plt.figure(figsize=(12, 10))
    plt.barh(categories, probabilities, color=colors)
    fontsize = 12

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Probability', fontsize=fontsize)
    plt.ylabel('DR Category', fontsize=fontsize)
    plt.title('Probability Distribution for Different Categories', fontsize=fontsize)
    plt.xlim(0, 1)  # Ensuring the x-axis ranges from 0 to 1

    # plt.show()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_format", type=str, default="*")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--mode", type=str, default="Classification", choices=["Classification", "ObjectDetection"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--resume", type=str)

    args = parser.parse_args()

    # chose fine-tuned model from 'checkpoint'
    model = prepare_ft_model(chkpt_dir=args.resume, num_classes=args.num_classes)

    # get all images
    # supports single files or folders
    data_path = args.data_path
    image_paths = [data_path] if os.path.isfile(data_path) else \
        [str(path) for path in pathlib.Path(data_path).rglob(f"*.{args.data_format}")]

    # run classification for each image
    for img_path in image_paths:
        run_classification(img_path=img_path, input_size=args.input_size, model=model)
