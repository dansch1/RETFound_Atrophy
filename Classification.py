import argparse

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn

import models_vit

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


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


def prepare_data():
    # load an image
    img = Image.open(args.data_path)
    img = img.resize((224, 224))
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


def run_classification(model):
    x = prepare_data()

    # model inference
    with torch.no_grad():
        output = model(x)

    output = nn.Softmax(dim=1)(output)
    output = output.squeeze(0).cpu().detach().numpy()

    print(f"Results: {output}")

    # visualization
    categories = ["Atrophy", "Normal"]
    colors = ["red", "green"]
    prob_result = draw_result(output, categories, colors)

    Image.fromarray(prob_result).save('classification.png')


def draw_result(probabilities, categories, colors):
    # Creating the bar plot
    fig = plt.figure(figsize=(12, 10))
    plt.barh(categories, probabilities, color=colors)
    if len(categories) == 39:
        fontsize = 8
    else:
        fontsize = 12
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Probability', fontsize=fontsize)
    plt.ylabel('DR Category', fontsize=fontsize)
    plt.title('Probability Distribution for Different DR Categories', fontsize=fontsize)
    plt.xlim(0, 1)  # Ensuring the x-axis ranges from 0 to 1

    # plt.show()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--resume", type=str)
    args = parser.parse_args()

    # chose fine-tuned model from 'checkpoint'
    model = prepare_ft_model(chkpt_dir=args.resume, num_classes=args.num_classes)
    run_classification(model)
