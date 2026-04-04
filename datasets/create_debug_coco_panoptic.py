#!/usr/bin/env python3
# Creates a 1/few image version of the COCO panoptic dataset for debugging purposes
import json
import os

def create_new_ds(original_json, oneimage_json, image_ids_keep):
    """
    Create new 1/multi image dataset from existing COCO panoptic dataset.
    Args:
        oneimage_json (str): New JSON file to write
        panoptic_root (str): Existing JSON file to use as template
        image_ids_keep (list): List of image ids to keep
    """

    with open(original_json, 'r') as f:
        original = json.load(f)

    new_images = []
    new_annotations = []
    for i in original['images']:
        if i['id'] in image_ids_keep:
            new_images.append(i)

    for i in original['annotations']:
        if i['image_id'] in image_ids_keep:
            new_annotations.append(i)

    new_json = {'info': original['info'], 'licenses': original['licenses'], 'images': new_images, 'annotations': new_annotations, 'categories': original['categories']}
    new_json['info']['description'] += ', selected images'

    with open(oneimage_json, 'w') as f:
        json.dump(new_json, f)

    print(f'Wrote new JSON keeping only images {image_ids_keep} to {oneimage_json}')


if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco")
    split = "val2017"
    create_new_ds(os.path.join(dataset_dir, "annotations/panoptic_{}.json".format(split)),
                  os.path.join(dataset_dir, "annotations/single_image_panoptic_debug_{}.json".format(split)),
                  image_ids_keep=[435081]) # Chosen as it has decent content and some overlapping instances

    create_new_ds(os.path.join(dataset_dir, "annotations/panoptic_{}.json".format(split)),
                  os.path.join(dataset_dir, "annotations/multi_image_panoptic_debug_{}.json".format(split)),
                  image_ids_keep=[435081, 18380, 210273, 546219, 273711])

# Example code for finding COCO images with the most instances
# import json
#
# with open('/usr/local/data/msmith/APL/Datasets/coco/annotations/panoptic_val2017.json', 'r') as f:
#     original = json.load(f)
#
# annotation_dict = {}
# for i in original['annotations']:
#     img_id = i['image_id']
#     if img_id not in annotation_dict:
#         annotation_dict[img_id] = []
#     annotation_dict[img_id].extend(i['segments_info'])
#
# sorted_ids = sorted(annotation_dict, key=lambda x: len(annotation_dict[x]), reverse=True)
#
# for k in sorted_ids:
#     print(f'Image {k} has {len(annotation_dict[k])} annotations')
#
