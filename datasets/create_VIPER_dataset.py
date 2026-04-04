#!/usr/bin/env python3
import argparse
import json
import multiprocessing
import os
import shutil
import time
from functools import partial
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from panopticapi.utils import get_traceback, rgb2id, save_json
from pycocotools import mask as COCOmask
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map


def get_args():
    parser = argparse.ArgumentParser(description='Create COCO-formatted VIPER dataset')
    parser.add_argument('dataset_dir', type=str, help='Path to the VIPER dataset, downloaded and extracted')
    parser.add_argument('output_dir', type=str, help='Path to where to create the new COCO format VIPER dataset')
    return parser


# Modified from https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/cocostuffhelper.py#L130
def getCMap(stuffStartId=0, stuffEndId=22, cmapName='jet', addUnlabeled=True):
    '''
    Create a color map for the classes in the COCO Stuff Segmentation Challenge.
    :param stuffStartId: (optional) index where stuff classes start
    :param stuffEndId: (optional) index where stuff classes end
    :param cmapName: (optional) Matlab's name of the color map
    :param addUnlabeled: (optional) whether to add a color for the 'unlabeled' class
    :return: cmap - [c, 3] a color map for c colors where the columns indicate the RGB values
    '''

    # Get jet color map from Matlab
    labelCount = stuffEndId - stuffStartId + 1
    cmapGen = matplotlib.colormaps.get_cmap(cmapName)
    cmapGen = cmapGen.resampled(labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]

    # Reduce value/brightness of stuff colors (easier in HSV format)
    cmap = cmap.reshape((-1, 1, 3))
    hsv = matplotlib.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = matplotlib.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    # Permute entries to avoid classes with similar name having similar colors
    rng = np.random.default_rng()
    perm = rng.permutation(labelCount)
    cmap = cmap[perm, :]

    # Add black color for 'unlabeled' class
    if addUnlabeled:
        cmap = np.vstack(((0.0, 0.0, 0.0), cmap))

    return cmap


def getSeqFrameInfo(file):
    seq = file.parent.name
    parts = file.stem.split('_')
    assert parts[0] == seq, f'File {file} has sequence {parts[0]} but is in folder for sequence {seq}'
    frame = parts[1]
    name = file.name
    return frame, seq, name


def move_image(rgb: Path, new_rgb_dir: Path):
    seq = rgb.parent.name
    new_rgb_seq_dir = new_rgb_dir / seq
    new_rgb_seq_dir.mkdir(parents=True, exist_ok=True)
    return shutil.copy(rgb, new_rgb_seq_dir)


def getClasses(classes: pd.DataFrame, coco_format_class_list: list, old_id_to_new_id: dict, instances: int = 0,
               startIdx=1):
    i = startIdx
    for c in classes.iterrows():
        viper_id, trainid, instance_eval = c[1].id, c[1].trainid, c[1].instance_eval
        if trainid == 255 or instance_eval != instances:
            continue
        old_id_to_new_id[viper_id] = i
        coco_format_class_list.append({
            'id': i,
            'name': c[1].classname,
            'supercategory': 'None',
            'isthing': instance_eval
        })
        i += 1
    return i


def processDataset(dataset_dir: Path, output_dir: Path):
    class_file = dataset_dir / 'classes.csv'
    assert class_file.exists(), 'CSV class file from https://github.com/srrichter/viper/blob/master/classes.csv does not exist!'

    classes = pd.read_csv(class_file)

    # We reformat the classes such that class IDs go from 1 - N and the things classes are all first
    # This follows the coco_2017_val_panoptic_with_semseg scheme which is pretty specific
    coco_format_class_list = []
    old_id_to_new_id = {}
    startIdx = getClasses(classes, coco_format_class_list, old_id_to_new_id, instances=1)
    getClasses(classes, coco_format_class_list, old_id_to_new_id, instances=0, startIdx=startIdx)

    splits = ['train', 'val']

    cmap = getCMap()

    for split in splits:

        VIPER_split_dir = dataset_dir / split
        assert VIPER_split_dir.exists(), f'VIPER split directory {VIPER_split_dir} does not exist!'

        img_folder = VIPER_split_dir / Path('img')
        class_folder = VIPER_split_dir / Path('cls')
        instance_folder = VIPER_split_dir / Path('inst')

        # Get all mask and label images for all sequences
        all_rgb_images = list(img_folder.glob('*/*.jpg'))
        all_cls_labels = list(class_folder.glob('*/*.png'))
        all_inst_labels = list(instance_folder.glob('*/*.png'))

        assert len(all_rgb_images) > 0, f"No images found in directory {img_folder}!"
        assert len(all_cls_labels) > 0, f"No semantic labels found in directory {class_folder}!"
        assert len(all_inst_labels) > 0, f"No instance labels found in directory {instance_folder}!"
        assert len(all_cls_labels) == len(all_inst_labels), 'Must have same number of images and labels'

        all_files = all_rgb_images + all_cls_labels + all_inst_labels

        corrupted_files = []
        for file in tqdm(all_files, desc='Checking for existing corrupted files'):
            if file.stat().st_size == 0:
                corrupted_files.append(file)

        for c in corrupted_files:
            print(f'File {c} has size 0!')
        if len(corrupted_files) > 0:
            raise Exception(
                'Please fix corrupted files before converting the dataset. Note: imagemagick with -quality 77 is about equal to original quality if fixing corrupted JPG images from PNG.')

        panoptic_output_folder = output_dir / ('panoptic_' + split)
        panoptic_output_folder.mkdir(parents=True, exist_ok=True)

        class_output_folder = output_dir / ('semseg_' + split)
        class_output_folder.mkdir(parents=True, exist_ok=True)

        annotation_dir = output_dir / Path('annotations')
        annotation_dir.mkdir(parents=True, exist_ok=True)

        out_filename = annotation_dir / f'panoptic_coco_format_{split}.json'

        # Move RGB images
        new_rgb_dir = output_dir / split
        new_rgb_dir.mkdir(parents=True, exist_ok=True)

        image_list = thread_map(partial(move_image, new_rgb_dir=new_rgb_dir), all_rgb_images, desc='Moving RGB images')
        image_list_lookup = {Path(i).stem: i for i in image_list}

        data = {'info': {'description': 'VIPER dataset', 'url': 'https://playing-for-benchmarks.org',
                         'version': '2.0',
                         'year': 2024,
                         'contributor': 'VIPER authors, this format: Michael Smith @McGill University',
                         'date_created': '2024/12/24'},
                'images': [],
                'annotations': [],
                'categories': coco_format_class_list}

        # Process images and annotations
        data_image = []
        data_annotations = []
        image_id_counter = 1
        annotation_id_counter = 1

        for cls_img in tqdm(all_cls_labels, desc='Converting label data'):
            # Extract frame and sequence info from filename
            frame, seq, name = getSeqFrameInfo(cls_img)

            # Get image size
            rgb_img_fn = image_list_lookup[cls_img.stem]
            im = Image.open(rgb_img_fn)
            im_size = im.size
            im.close()

            # Basic image data for panoptic JSON
            data_image.append({
                'id': image_id_counter,
                'width': im_size[0],
                'height': im_size[1],
                'file_name': str(Path(seq) / name.replace('.png', '.jpg')),
                # Path to RGB image is relative to image root dir
                'sequence': int(seq),
                'frameID': int(frame)
            })

            # Write semantic-only PNG
            semantic_data = np.asarray(Image.open(cls_img))
            new_semantic_data = np.full_like(semantic_data, 255)  # 255 is ignore/void typically
            # PNG semantic data for this COCO dataset configuration is expected to be equal to the position in the
            # class list, so effectively class ID - 1
            for cl in classes.iterrows():
                id, trainId = cl[1].id, cl[1].trainid
                if trainId == 255:
                    continue
                old_region = semantic_data == id
                if old_region.sum().item() > 0:
                    new_semantic_data[old_region] = old_id_to_new_id[id] - 1

            new_inst_seq_folder = class_output_folder / seq
            new_inst_seq_folder.mkdir(parents=True, exist_ok=True)

            png = Image.fromarray(new_semantic_data).convert('P')
            png.putpalette(cmap)
            png.save(new_inst_seq_folder / name, format='PNG')

            inst_img_fn = instance_folder / seq / name
            instance_data = np.asarray(Image.open(inst_img_fn), dtype=np.uint32)

            # Panoptic annotations
            # # of annotations = # of images
            # Each annotation entry has "segments_info" dict key which contains all objects in image
            # VIPER has two PNG files per image: one for class labes and one for instance segmentation
            # The instance image non-zero pixels only cover instance-type objects. E.g. Sky is a valid class
            # for a pixel in the class label image, but the same pixel will be [0,0,0] in the instance once

            # Determine number of unique semantic instances
            semantic_class_ids = instance_data[:, :, 0]
            semantic_instance_ids = 256 * instance_data[:, :, 1] + instance_data[:, :, 2]
            unique_semantic_instances = np.unique(semantic_instance_ids)

            local_panoptic_output_folder = panoptic_output_folder / seq
            local_panoptic_output_folder.mkdir(parents=True, exist_ok=True)

            im_annotation = {'image_id': image_id_counter,
                             'file_name': str(
                                 Path(seq) / name)}  # Path to panoptic image is relative to panoptic label dir

            segments_info = []
            panoptic_image = np.zeros_like(new_semantic_data, dtype=np.uint32)  # For panoptic 0 is void
            # Process instances first
            # Ignore 0 - these are areas with no instances
            for inst in unique_semantic_instances:
                if inst == 0:
                    continue
                mask = semantic_instance_ids == inst
                mask_class_id = semantic_class_ids[mask]
                assert (mask_class_id == mask_class_id[0]).all()
                class_row = classes.loc[mask_class_id[0]]
                trainId, id = class_row.trainid, class_row.id
                if trainId == 255:
                    continue

                segments_info.append({
                    'id': annotation_id_counter,
                    'category_id': old_id_to_new_id[id],
                    'objID': int(inst),
                    'iscrowd': 0,
                    'area': mask.sum().item()
                })
                panoptic_image[mask] = annotation_id_counter
                annotation_id_counter += 1

            # Process remaining semantic regions
            for cl in classes.iterrows():
                id, trainId = cl[1].id, cl[1].trainid
                if trainId == 255:
                    continue
                # We only assign pixels in regions that are not already covered by the instance image
                semantic_region = (semantic_data == id) & (semantic_class_ids == 0)
                if semantic_region.sum().item() > 0:
                    segments_info.append({
                        'id': annotation_id_counter,
                        'category_id': old_id_to_new_id[id],
                        'objID': -1,
                        'iscrowd': 0,
                        'area': semantic_region.sum().item()
                    })
                    panoptic_image[semantic_region] = annotation_id_counter
                    annotation_id_counter += 1

            im_annotation['segments_info'] = segments_info
            data_annotations.append(im_annotation)

            # Save panoptic image
            pan_format = np.zeros(instance_data.shape, dtype=np.uint8)
            pan_format[:, :, 0] = panoptic_image % 256
            pan_format[:, :, 1] = panoptic_image // 256
            pan_format[:, :, 2] = panoptic_image // 256 // 256
            Image.fromarray(pan_format).save(local_panoptic_output_folder / name)

            image_id_counter += 1

        data['images'] = data_image
        data['annotations'] = data_annotations

        with open(out_filename, 'w') as f:
            json.dump(data, f)

        # Convert panoptic to detection format for completeness and, depending on implementation, ability to load COCO at all
        convert_panoptic_to_detection_coco_format(out_filename,
                                                  panoptic_output_folder,
                                                  annotation_dir / f'instances_coco_format_{split}.json',
                                                  True)


# Below adapted from https://github.com/cocodataset/panopticapi/blob/master/converters/panoptic2detection_coco_format.py
# It converts panoptic COCO format to detection COCO format
@get_traceback
def convert_panoptic_to_detection_coco_format_single_core(
        proc_id, annotations_set, categories, segmentations_folder, things_only
):
    annotations_detection = []
    for working_idx, annotation in enumerate(annotations_set):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id,
                                                                 working_idx,
                                                                 len(annotations_set)))

        file_name = '{}.png'.format(annotation['file_name'].rsplit('.')[0])
        try:
            pan_format = np.array(
                Image.open(os.path.join(segmentations_folder, file_name)), dtype=np.uint32
            )
        except IOError:
            raise KeyError('no prediction png file for id: {}'.format(annotation['image_id']))
        pan = rgb2id(pan_format)

        for segm_info in annotation['segments_info']:
            if things_only and categories[segm_info['category_id']]['isthing'] != 1:
                continue
            mask = (pan == segm_info['id']).astype(np.uint8)
            mask = np.expand_dims(mask, axis=2)
            segm_info.pop('id')
            segm_info['image_id'] = annotation['image_id']
            rle = COCOmask.encode(np.asfortranarray(mask))[0]
            rle['counts'] = rle['counts'].decode('utf8')
            segm_info['segmentation'] = rle
            annotations_detection.append(segm_info)

    print('Core: {}, all {} images processed'.format(proc_id, len(annotations_set)))
    return annotations_detection


def convert_panoptic_to_detection_coco_format(input_json_file,
                                              segmentations_folder,
                                              output_json_file,
                                              things_only):
    start_time = time.time()

    if segmentations_folder is None:
        segmentations_folder = input_json_file.rsplit('.', 1)[0]

    print("CONVERTING...")
    print("COCO panoptic format:")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(input_json_file))
    print("TO")
    print("COCO detection format")
    print("\tJSON file: {}".format(output_json_file))
    if things_only:
        print("Saving only segments of things classes.")
    print('\n')

    print("Reading annotation information from {}".format(input_json_file))
    with open(input_json_file, 'r') as f:
        d_coco = json.load(f)
    annotations_panoptic = d_coco['annotations']

    categories_list = d_coco['categories']
    categories = {category['id']: category for category in categories_list}

    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(annotations_panoptic, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotations_set in enumerate(annotations_split):
        p = workers.apply_async(convert_panoptic_to_detection_coco_format_single_core,
                                (proc_id, annotations_set, categories, segmentations_folder, things_only))
        processes.append(p)
    annotations_coco_detection = []
    for p in processes:
        annotations_coco_detection.extend(p.get())
    for idx, ann in enumerate(annotations_coco_detection):
        ann['id'] = idx

    d_coco['annotations'] = annotations_coco_detection
    categories_coco_detection = []
    for category in d_coco['categories']:
        if things_only and category['isthing'] != 1:
            continue
        category.pop('isthing')
        categories_coco_detection.append(category)
    d_coco['categories'] = categories_coco_detection
    save_json(d_coco, output_json_file)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    args = get_args().parse_args()

    dataset_dir = Path(args.dataset_dir)
    assert dataset_dir.is_dir(), f"Dataset folder {dataset_dir} does not exist"

    output_dir = Path(args.output_dir)

    processDataset(dataset_dir, output_dir)
