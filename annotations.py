import json
import os

from lxml import etree

FILE_PATH = r"C:\Users\d_schr33\Desktop\Unterlagen\Promotion\Data\train_data"
ANNOTATIONS = r"C:\Users\d_schr33\Desktop\Unterlagen\Promotion\Data\cam 2.0\annotations.xml"
OUTPUT_FILE = r"annotations.json"

# CLASS_TO_IDX = {"iORA": 1, "cORA": 1, "iRORA": 1, "cRORA": 1}  # 1 class: 1 - atrophy
CLASS_TO_IDX = {"iORA": 1, "cORA": 1, "iRORA": 2, "cRORA": 2}  # 2 classes: 1 - iORA + cORA, 2 - iRORA + cRORA


# CLASS_TO_IDX = {"iORA": 1, "cORA": 2, "iRORA": 3, "cRORA": 4}  # 4 classes: 1 - iORA, 2 - cORA, 3 - iRORA, 4 - cRORA


def load_images():
    result = []

    for path, subdirs, files in os.walk(FILE_PATH):
        for name in files:
            result.append((path, name))

    return result


def load_annotations():
    return etree.parse(ANNOTATIONS).getroot()


def load_intervals(images, annotations):
    result = {}

    for path, name in images:
        bboxes = annotations.findall(f"image[@name='{name}']/box")
        intervals = []

        for bbox in bboxes:
            # get start and end point
            x0, x1 = float(bbox.get("xtl")), float(bbox.get("xbr"))
            cls = [attribute.get("name") for attribute in bbox.findall("attribute") if attribute.text == "true"]

            if len(cls) == 1:
                intervals.append([x0, x1, CLASS_TO_IDX[cls[0]]])

        result[name] = combine_intervals(intervals)

    return result


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


def create_json(images, intervals):
    result = []

    for path, name in images:
        result.append({"image_path": os.path.join(path, name), "targets": intervals.get(name, [])})

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    images = load_images()
    annotations = load_annotations()
    intervals = load_intervals(images, annotations)

    create_json(images, intervals)
