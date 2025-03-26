import xmltodict
from functools import reduce
import argparse


def training_data_info(annotations):
    with open(annotations, "r") as file:
        data = file.read()

    images = xmltodict.parse(data)["annotations"]["image"]

    patients = {}
    classes_total = {"cRORA": 0, "iORA": 0, "cORA": 0, "iRORA": 0}
    classes_patient = {"cRORA": {}, "iORA": {}, "cORA": {}, "iRORA": {}}
    normal_images = []

    for image in images:
        # count patients and images
        name = image["@name"].replace(" ", ".").split(".")

        id = name[0]
        index = name[1] if len(name) > 2 else "-1"

        patients.setdefault(id, []).append(index)

        if "box" not in image:
            normal_images.append(image["@name"])
            continue

        # count classes
        atrophy = image["box"]

        if type(atrophy) is not list:
            atrophy = [atrophy]

        for bbox in atrophy:
            for attribute in bbox["attribute"]:
                if attribute["#text"] == "true":
                    classes_total[attribute["@name"]] += 1

                hits = classes_patient[attribute["@name"]]
                hits[id] = max(hits[id] if id in hits else 0, 1 if attribute["#text"] == "true" else 0)

    print(f"Patients: {len(patients.keys())} "
          f"with {reduce(lambda count, l: count + len(l), patients.values(), 0)} images.")
    print(f"Classes total: {classes_total}")
    print(classes_patient)
    print(normal_images)

    for k, v in classes_patient.items():
        print(f"{k}: {sum(v.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str)
    args = parser.parse_args()

    training_data_info(args.annotations)
