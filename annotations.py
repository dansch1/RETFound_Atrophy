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


def combine_annotations(images, annotations, num_classes, x_offset):
    result = {}

    for path, name in images:
        class_intervals = get_class_intervals(image=name, annotations=annotations, num_classes=num_classes)

        if len(class_intervals) == 0:
            continue

        result[name] = [[max(x0 - x_offset, 0), max(x1 - x_offset, 0), cls] for x0, x1, cls in class_intervals]

    return result


def get_class_intervals(image, annotations, num_classes):
    bboxes = annotations.findall(f"image[@name='{image}']/box")
    intervals = []

    if len(bboxes) == 0:
        return []

    for bbox in bboxes:
        # get start and end point
        try:
            x0 = float(bbox.get("xtl"))
            x1 = float(bbox.get("xbr"))
        except (TypeError, ValueError):
            print(f"Invalid coordinates in image {image}")
            continue

        attrs = bbox.findall("attribute")

        if not attrs:
            raw_cls = bbox.get("label")

            if raw_cls == "no class":
                return []

        else:
            true_attrs = [a.get("name") for a in attrs if (a.text or "").strip().lower() == "true"]

            if len(true_attrs) != 1:
                print(f"Found illegal attributes for image {image}")
                continue

            raw_cls = true_attrs[0]

        if raw_cls not in CLASS_REDUCTION[num_classes].keys():
            print(f"Could not parse class for image {image}")
            continue

        cls = CLASS_REDUCTION[num_classes][raw_cls]
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


def save_annotations(annotations, output_path, prefix):
    dir_name = os.path.dirname(output_path)
    base_name = os.path.basename(output_path)
    new_base_name = prefix + base_name
    new_path = os.path.join(dir_name, new_base_name)

    with open(new_path, "w") as f:
        json.dump(annotations, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--annotations", type=str)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--x_offset", type=int, default=496)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    images = load_images(args.data_path)
    annotations = load_annotations(args.annotations)

    new_annotations = combine_annotations(images=images, annotations=annotations, num_classes=args.num_classes,
                                          x_offset=args.x_offset)
    save_annotations(annotations=new_annotations, output_path=args.output_path, prefix=f"{args.num_classes}c_")
