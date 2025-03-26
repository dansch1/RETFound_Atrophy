import json
import os

from lxml import etree
import argparse

CLASS_REDUCTION = {2: {"iORA": 1, "cORA": 1, "iRORA": 1, "cRORA": 1},  # 1 class: (0.Normal), 1.atrophy
                   3: {"iORA": 1, "cORA": 1, "iRORA": 2, "cRORA": 2},  # 2 classes: 1.iORA+cORA, 2.iRORA+cRORA
                   5: {"iORA": 1, "cORA": 2, "iRORA": 3, "cRORA": 4}}  # 4 classes: 1.iORA, 2.cORA, 3.iRORA, 4.cRORA


def load_images(data_path):
    result = []

    for path, subdirs, files in os.walk(data_path):
        for name in files:
            result.append((path, name))

    return result


def load_annotations(annotations):
    return etree.parse(annotations).getroot()


def get_targets(images, annotations, num_classes):
    result = {}

    for path, name in images:
        result[name] = get_class_intervals(image=name, annotations=annotations, num_classes=num_classes)

    return result


def get_class_intervals(image, annotations, num_classes):
    bboxes = annotations.findall(f"image[@name='{image}']/box")
    intervals = []

    for bbox in bboxes:
        # get start and end point
        x0, x1 = float(bbox.get("xtl")), float(bbox.get("xbr"))
        cls = [attribute.get("name") for attribute in bbox.findall("attribute") if attribute.text == "true"]

        if len(cls) == 1:
            cls = CLASS_REDUCTION[num_classes][cls[0]]
            intervals.append([x0, x1, cls])

    return combine_intervals(intervals)


def combine_intervals(intervals):
    if len(intervals) <= 1:
        return intervals

    sorted_intervals = sorted(intervals, key=lambda l: l[0])  # strictly monotonically increasing
    result = [sorted_intervals[0]]

    for i in range(1, len(sorted_intervals)):
        if result[-1][2] != sorted_intervals[i][2]:
            result.append(sorted_intervals[i])
            continue

        if result[-1][1] >= sorted_intervals[i][1]:
            continue

        if abs(result[-1][1] - sorted_intervals[i][0]) < 10:
            result[-1][1] = sorted_intervals[i][1]
        else:
            result.append(sorted_intervals[i])

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--annotations", type=str)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--output_path", type=str, default="annotations.json")
    args = parser.parse_args()

    images = load_images(args.data_path)
    annotations = load_annotations(args.annotations)
    targets = get_targets(images=images, annotations=annotations, num_classes=args.num_classes)

    with open(args.output_path, "w") as f:
        json.dump(targets, f, indent=4)
