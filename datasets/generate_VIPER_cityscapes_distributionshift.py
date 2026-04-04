#!/usr/bin/env python3
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from cityscapesscripts.helpers.labels import id2label as cityscapes_id_to_label
from panopticapi.utils import rgb2id, id2rgb
from tqdm.auto import tqdm

from create_VIPER_dataset import convert_panoptic_to_detection_coco_format

# Note: this is semantic ID (position in panoptic category list) VIPER to ID Cityscapes
# 255 is void for VIPER, 0 is unlabeled (effectively void) for Cityscapes
VIPERToCityscapesLabelMap = {
    0: 19,  # trafficlight -> traffic light
    1: 0,  # firehydrant -> void
    2: 0,  # chair -> void
    3: 0,  # trashcan -> void
    4: 24,  # person -> person
    5: 32,  # motorcycle -> motorcycle
    6: 26,  # car -> car
    7: 26,  # van -> car (included in Cityscapes definition)
    8: 28,  # bus -> bus
    9: 27,  # truck -> truck
    10: 23,  # sky -> sky
    11: 7,  # road -> road
    12: 8,  # sidewalk -> sidewalk
    13: 22,  # terrain -> terrain
    14: 21,  # tree -> vegetation (trees included by definition)
    15: 21,  # vegetation -> vegetation
    16: 11,  # building -> building
    17: 0,  # infrastructure -> void (infrastructure includes walls, overpasses, street lights...)
    18: 13,  # fence -> fence
    19: 0,  # billboard (ad part only, not supporting structure) -> void
    20: 20,  # trafficsign -> traffic sign
    21: 0,  # mobilebarrier (construction barriers) -> void
    22: 0,  # trash (trash bags, discarded food containers...) -> void
    255: 0  # void -> void
}

VIPERPanopticToCityscapesLabelMap = {i + 1: j for i, j in VIPERToCityscapesLabelMap.items()}

VIPERToCityscapesTrainLabelMap = {i: cityscapes_id_to_label[j].trainId for i, j in VIPERToCityscapesLabelMap.items()}

# Cityscapes ID to VIPER semantic ID
CityscapesToVIPERLabelMap = {
    7: 11,  # road -> road
    8: 12,  # sidewalk -> sidewalk
    11: 16,  # building -> building
    12: 17,  # wall -> infrastructure
    13: 18,  # fence -> fence
    17: 17,  # pole -> infrastructure
    19: 0,  # traffic light -> trafficlight
    20: 20,  # traffic sign -> trafficsign
    21: 15,  # vegetation -> vegetation
    22: 13,  # terrain-> terrain
    23: 10,  # sky -> sky
    24: 4,  # person -> person
    25: 4,  # rider -> person
    26: 6,  # car -> car
    27: 9,  # truck -> truck
    28: 8,  # bus -> bus
    31: 255,  # train -> void
    32: 5,  # motorcycle -> motorcycle
    33: 255,  # bicycle -> void
    0: 255  # void -> void
}

CityscapesToVIPERPanopticLabelMap = {i: j + 1 for i, j in CityscapesToVIPERLabelMap.items()}


def copyFlowVIPERToCity(viper_flow_path: Path, city_flow_path: Path, city: str):
    viper_flows = list(viper_flow_path.glob('*/*.npz'))
    for v_flow in tqdm(viper_flows, desc='Copying optical flow'):
        seq, frame_no = v_flow.stem.split('_')
        suffix = v_flow.suffix
        new_flow_file = city_flow_path / city / f'{city}_{seq:0>6}_{frame_no:0>6}_leftImg8bit{suffix}'
        new_flow_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(v_flow, new_flow_file)


def copyFlowCityToVIPER(city_flow_path: Path, viper_flow_path: Path):
    city_flows = list(city_flow_path.glob('*/*.npz'))
    for c_flow in tqdm(city_flows, desc='Copying optical flow'):
        city, seq, frame_no, file_type = c_flow.stem.split('_')
        suffix = c_flow.suffix
        seq, frame_no = int(seq), int(frame_no)
        new_flow_file = viper_flow_path / f'{city}' / f'{city}_{seq:0>6}_{frame_no:0>6}{suffix}'
        new_flow_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c_flow, new_flow_file)


def processVIPER(dataset_dir: Path):
    viper_dir = dataset_dir / 'VIPER'
    output_dir = dataset_dir / 'VIPER_cityscapesformat'
    print(f'Processing VIPER dataset at {viper_dir}')
    print(f'Writing to {output_dir}')

    city = 'GTA5'

    for split in ['train', 'val']:
        print(f'Processing split {split}')
        semseg_dir = viper_dir / ('semseg_' + split)
        panoptic_dir = viper_dir / ('panoptic_' + split)
        cityscapes_semseg_dir = output_dir / 'gtFine' / split
        cityscapes_semseg_dir.mkdir(parents=True, exist_ok=True)
        file_list = list(semseg_dir.glob('*/*.png'))

        for file in tqdm(file_list, desc='Processing semantic annotations'):
            data = np.asarray(Image.open(file))
            new_id_map = np.zeros_like(data)
            new_trainID_map = np.zeros_like(data)
            for viper_cat, city_cat in VIPERToCityscapesLabelMap.items():
                new_id_map[data == viper_cat] = city_cat
            for viper_cat, city_cat in VIPERToCityscapesTrainLabelMap.items():
                new_trainID_map[data == viper_cat] = city_cat
            seq, frame_no = file.stem.split('_')
            seq, frame_no = int(seq), int(frame_no)

            id_fn = cityscapes_semseg_dir / city / f'{city}_{seq:0>6}_{frame_no:0>6}_gtFine_labelIds.png'
            trainid_fn = cityscapes_semseg_dir / city / f'{city}_{seq:0>6}_{frame_no:0>6}_gtFine_labelTrainIds.png'
            id_fn.parent.mkdir(parents=True, exist_ok=True)
            trainid_fn.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(new_id_map).save(id_fn, format='PNG')
            Image.fromarray(new_trainID_map).save(trainid_fn, format='PNG')

        viper_rgb_folder = viper_dir / split
        cityscapes_rgb_folder = output_dir / 'leftImg8bit' / split
        for rgb_file in tqdm(list(viper_rgb_folder.glob('*/*.jpg')), desc='Copying RGB files'):
            seq, frame_no = rgb_file.stem.split('_')
            suffix = rgb_file.suffix
            seq, frame_no = int(seq), int(frame_no)
            new_rgb_file = cityscapes_rgb_folder / city / f'{city}_{seq:0>6}_{frame_no:0>6}_leftImg8bit{suffix}'
            new_rgb_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rgb_file, new_rgb_file)

        if split == 'val':
            copyFlowVIPERToCity(viper_dir / 'optflow_extract' / split / 'flow', output_dir / 'optflow_fw' / split, city)
            copyFlowVIPERToCity(viper_dir / 'optflow_extract' / split / 'flowbw', output_dir / 'optflow_bw' / split,
                                city)

        # Deal with panoptic
        with open(viper_dir / 'annotations' / ('panoptic_coco_format_' + split + '.json'), 'r') as f:
            panoptic_json = json.load(f)

        new_json = {'info': {'description': 'VIPER dataset - Cityscapes label format',
                             'url': 'https://playing-for-benchmarks.org',
                             'version': '2.0',
                             'year': 2025,
                             'contributor': 'VIPER authors, this format: Michael Smith @McGill University',
                             'date_created': '2025/03/05'},
                    'categories': [{'color': [128, 64, 128],
                                    'id': 7,
                                    'isthing': 0,
                                    'name': 'road',
                                    'supercategory': 'flat',
                                    'ignore': False},
                                   # Ignore field = ignore in eval as not mapped and thus no ground truth
                                   {'color': [244, 35, 232],
                                    'id': 8,
                                    'isthing': 0,
                                    'name': 'sidewalk',
                                    'supercategory': 'flat',
                                    'ignore': False},
                                   {'color': [70, 70, 70],
                                    'id': 11,
                                    'isthing': 0,
                                    'name': 'building',
                                    'supercategory': 'construction',
                                    'ignore': False},
                                   {'color': [102, 102, 156],
                                    'id': 12,
                                    'isthing': 0,
                                    'name': 'wall',
                                    'supercategory': 'construction',
                                    'ignore': True},
                                   {'color': [190, 153, 153],
                                    'id': 13,
                                    'isthing': 0,
                                    'name': 'fence',
                                    'supercategory': 'construction',
                                    'ignore': False},
                                   {'color': [153, 153, 153],
                                    'id': 17,
                                    'isthing': 0,
                                    'name': 'pole',
                                    'supercategory': 'object',
                                    'ignore': True},
                                   {'color': [250, 170, 30],
                                    'id': 19,
                                    'isthing': 0,
                                    'name': 'traffic light',
                                    'supercategory': 'object',
                                    'ignore': False},
                                   {'color': [220, 220, 0],
                                    'id': 20,
                                    'isthing': 0,
                                    'name': 'traffic sign',
                                    'supercategory': 'object',
                                    'ignore': False},
                                   {'color': [107, 142, 35],
                                    'id': 21,
                                    'isthing': 0,
                                    'name': 'vegetation',
                                    'supercategory': 'nature',
                                    'ignore': False},
                                   {'color': [152, 251, 152],
                                    'id': 22,
                                    'isthing': 0,
                                    'name': 'terrain',
                                    'supercategory': 'nature',
                                    'ignore': False},
                                   {'color': [70, 130, 180],
                                    'id': 23,
                                    'isthing': 0,
                                    'name': 'sky',
                                    'supercategory': 'sky',
                                    'ignore': False},
                                   {'color': [220, 20, 60],
                                    'id': 24,
                                    'isthing': 1,
                                    'name': 'person',
                                    'supercategory': 'human',
                                    'ignore': False},
                                   {'color': [255, 0, 0],
                                    'id': 25,
                                    'isthing': 1,
                                    'name': 'rider',
                                    'supercategory': 'human',
                                    'ignore': True},
                                   {'color': [0, 0, 142],
                                    'id': 26,
                                    'isthing': 1,
                                    'name': 'car',
                                    'supercategory': 'vehicle',
                                    'ignore': False},
                                   {'color': [0, 0, 70],
                                    'id': 27,
                                    'isthing': 1,
                                    'name': 'truck',
                                    'supercategory': 'vehicle',
                                    'ignore': False},
                                   {'color': [0, 60, 100],
                                    'id': 28,
                                    'isthing': 1,
                                    'name': 'bus',
                                    'supercategory': 'vehicle',
                                    'ignore': False},
                                   {'color': [0, 80, 100],
                                    'id': 31,
                                    'isthing': 1,
                                    'name': 'train',
                                    'supercategory': 'vehicle',
                                    'ignore': True},
                                   {'color': [0, 0, 230],
                                    'id': 32,
                                    'isthing': 1,
                                    'name': 'motorcycle',
                                    'supercategory': 'vehicle',
                                    'ignore': False},
                                   {'color': [119, 11, 32],
                                    'id': 33,
                                    'isthing': 1,
                                    'name': 'bicycle',
                                    'supercategory': 'vehicle',
                                    'ignore': True}]}

        new_img_list = []
        for img in panoptic_json['images']:
            fn = Path(img['file_name'])
            seq, frame_no = fn.stem.split('_')
            seq, frame_no = int(seq), int(frame_no)
            suffix = fn.suffix
            new_rgb_file = f'{city}_{seq:0>6}_{frame_no:0>6}_leftImg8bit{suffix}'
            new_img = img.copy()
            new_img['file_name'] = new_rgb_file
            new_img['id'] = f'{city}_{seq:0>6}_{frame_no:0>6}'
            new_img_list.append(new_img)

        new_json['images'] = new_img_list

        new_panoptic_dir = output_dir / 'gtFine' / ('cityscapes_panoptic_' + split)
        ann_id_counter = 1
        new_annotation_list = []
        for ann in tqdm(panoptic_json['annotations'], desc='Processing panoptic and instance annotations'):
            per_category_instance_id = defaultdict(int)
            old_fn = Path(ann['file_name'])
            seq, frame_no = old_fn.stem.split('_')
            seq, frame_no = int(seq), int(frame_no)
            new_fn = f'{city}_{seq:0>6}_{frame_no:0>6}_gtFine_panoptic.png'
            new_inst_fn = f'{city}_{seq:0>6}_{frame_no:0>6}_gtFine_instanceIds.png'
            new_ann = {'image_id': f'{city}_{seq:0>6}_{frame_no:0>6}',
                       'file_name': new_fn}
            old_panoptic_mask = np.asarray(Image.open(panoptic_dir / old_fn))
            old_panoptic_mask = rgb2id(old_panoptic_mask)
            new_panoptic_mask = np.zeros_like(old_panoptic_mask)
            # Need to create instance annotations as well for cityscapes semantic eval to run
            new_instance_mask = np.zeros_like(new_panoptic_mask, dtype=np.int32)  # 0 = void for cityscapes
            seg_info = []
            for seg in ann['segments_info']:
                old_ann_id = seg['id']
                new_seg = seg.copy()
                new_cat_id = VIPERPanopticToCityscapesLabelMap[seg['category_id']]
                is_instance = cityscapes_id_to_label[new_cat_id].hasInstances
                if new_cat_id != 0:
                    new_seg['category_id'] = new_cat_id
                    new_seg['id'] = ann_id_counter
                    old_region = old_panoptic_mask == old_ann_id
                    new_panoptic_mask[old_region] = ann_id_counter
                    if is_instance:
                        new_instance_mask[old_region] = new_cat_id * 1000 + per_category_instance_id[new_cat_id]
                        per_category_instance_id[new_cat_id] += 1
                    else:
                        new_instance_mask[old_region] = new_cat_id
                    ann_id_counter += 1
                    seg_info.append(new_seg)
            new_panoptic_mask = id2rgb(new_panoptic_mask)

            new_file = new_panoptic_dir / new_fn
            new_file.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(new_panoptic_mask).save(new_file)

            new_inst_file = cityscapes_semseg_dir / city / new_inst_fn
            new_inst_file.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(new_instance_mask, mode="I").save(new_inst_file)

            new_ann['segments_info'] = seg_info

            new_annotation_list.append(new_ann)

        new_json['annotations'] = new_annotation_list

        panoptic_json_fn = output_dir / 'gtFine' / ('cityscapes_panoptic_' + split + '.json')
        with open(panoptic_json_fn, 'w') as f:
            json.dump(new_json, f)
        print(f'Wrote panoptic JSON annotations to {panoptic_json_fn}')


def processCityscapes(dataset_dir: Path):
    cityscapes_dir = dataset_dir / 'cityscapes'
    output_dir = dataset_dir / 'cityscapes_VIPERformat'
    print(f'Processing Cityscapes dataset at {cityscapes_dir}')
    print(f'Writing to {output_dir}')

    for split in ['train', 'val', 'test']:
        print(f'Processing split {split}')

        cityscapes_semseg_dir = cityscapes_dir / 'gtFine' / split
        cityscapes_panoptic_dir = cityscapes_dir / 'gtFine' / ('cityscapes_panoptic_' + split)
        viper_semseg_dir = output_dir / ('semseg_' + split)
        viper_semseg_dir.mkdir(parents=True, exist_ok=True)
        file_list = list(cityscapes_semseg_dir.glob('*/*gtFine_labelIds.png'))

        for file in tqdm(file_list, desc='Processing semantic annotations'):
            data = np.asarray(Image.open(file))
            new_id_map = np.full_like(data, 255)

            for city_cat, viper_cat in CityscapesToVIPERLabelMap.items():
                new_id_map[data == city_cat] = viper_cat
            city, seq, frame_no, _, file_type = file.stem.split('_')
            seq, frame_no = int(seq), int(frame_no)

            id_fn = viper_semseg_dir / f'{city}' / f'{city}_{seq:0>6}_{frame_no:0>6}.png'
            id_fn.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(new_id_map).save(id_fn, format='PNG')

        viper_rgb_folder = output_dir / split
        cityscapes_rgb_folder = cityscapes_dir / 'leftImg8bit' / split
        for rgb_file in tqdm(list(cityscapes_rgb_folder.glob('*/*.png')), desc='Copying RGB files'):
            city, seq, frame_no, file_type = rgb_file.stem.split('_')
            suffix = rgb_file.suffix
            seq, frame_no = int(seq), int(frame_no)
            new_rgb_file = viper_rgb_folder / f'{city}' / f'{city}_{seq:0>6}_{frame_no:0>6}{suffix}'
            new_rgb_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rgb_file, new_rgb_file)

        if split in ('val', 'train'):
            viper_rgb_ood_folder = output_dir / (split + '_ood')
            cityscapes_ood_rgb_folder = cityscapes_dir / 'leftImg8bit_ood' / split
            for rgb_file in tqdm(list(cityscapes_ood_rgb_folder.glob('*/*.png')), desc='Copying OOD RGB files'):
                city, seq, frame_no, file_type = rgb_file.stem.split('_')
                suffix = rgb_file.suffix
                seq, frame_no = int(seq), int(frame_no)
                new_rgb_file = viper_rgb_ood_folder / f'{city}' / f'{city}_{seq:0>6}_{frame_no:0>6}{suffix}'
                new_rgb_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rgb_file, new_rgb_file)

        cityscapes_rgb_sequence_folder = cityscapes_dir / 'leftImg8bit_sequence' / split
        if cityscapes_rgb_sequence_folder.is_dir():
            for rgb_file in tqdm(list(cityscapes_rgb_sequence_folder.glob('*/*.png')),
                                 desc='Copying RGB sequence files'):
                city, seq, frame_no, file_type = rgb_file.stem.split('_')
                suffix = rgb_file.suffix
                seq, frame_no = int(seq), int(frame_no)
                new_rgb_file = viper_rgb_folder / f'{city}' / f'{city}_{seq:0>6}_{frame_no:0>6}{suffix}'
                if not new_rgb_file.is_file():
                    new_rgb_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(rgb_file, new_rgb_file)

        if split == 'val':
            copyFlowCityToVIPER(cityscapes_dir / 'optflow_fw' / split, output_dir / 'optflow_extract' / split / 'flow')
            copyFlowCityToVIPER(cityscapes_dir / 'optflow_bw' / split,
                                output_dir / 'optflow_extract' / split / 'flowbw')

        with open(cityscapes_dir / 'gtFine' / ('cityscapes_panoptic_' + split + '.json'), 'r') as f:
            panoptic_json = json.load(f)

            new_json = {'info': {'description': 'Cityscapes dataset - VIPER label format',
                                 'url': 'https://www.cityscapes-dataset.com/',
                                 'version': '1.0',
                                 'year': 2025,
                                 'contributor': 'Cityscapes authors, this format: Michael Smith @McGill University',
                                 'date_created': '2025/03/05'},
                        'categories': [{'id': 1,
                                        'name': 'trafficlight',
                                        'supercategory': 'None',
                                        'isthing': 1,
                                        'ignore': False},
                                       {'id': 2,
                                        'name': 'firehydrant',
                                        'supercategory': 'None',
                                        'isthing': 1,
                                        'ignore': True},
                                       {'id': 3,
                                        'name': 'chair',
                                        'supercategory': 'None',
                                        'isthing': 1,
                                        'ignore': True},
                                       {'id': 4,
                                        'name': 'trashcan',
                                        'supercategory': 'None',
                                        'isthing': 1,
                                        'ignore': True},
                                       {'id': 5,
                                        'name': 'person',
                                        'supercategory': 'None',
                                        'isthing': 1,
                                        'ignore': False},
                                       {'id': 6,
                                        'name': 'motorcycle',
                                        'supercategory': 'None',
                                        'isthing': 1,
                                        'ignore': False},
                                       {'id': 7,
                                        'name': 'car',
                                        'supercategory': 'None',
                                        'isthing': 1,
                                        'ignore': False},
                                       {'id': 8,
                                        'name': 'van',
                                        'supercategory': 'None',
                                        'isthing': 1,
                                        'ignore': True},
                                       {'id': 9,
                                        'name': 'bus',
                                        'supercategory': 'None',
                                        'isthing': 1,
                                        'ignore': False},
                                       {'id': 10,
                                        'name': 'truck',
                                        'supercategory': 'None',
                                        'isthing': 1,
                                        'ignore': False},
                                       {'id': 11,
                                        'name': 'sky',
                                        'supercategory': 'None',
                                        'isthing': 0,
                                        'ignore': False},
                                       {'id': 12,
                                        'name': 'road',
                                        'supercategory': 'None',
                                        'isthing': 0,
                                        'ignore': False},
                                       {'id': 13,
                                        'name': 'sidewalk',
                                        'supercategory': 'None',
                                        'isthing': 0,
                                        'ignore': False},
                                       {'id': 14,
                                        'name': 'terrain',
                                        'supercategory': 'None',
                                        'isthing': 0,
                                        'ignore': False},
                                       {'id': 15,
                                        'name': 'tree',
                                        'supercategory': 'None',
                                        'isthing': 0,
                                        'ignore': True},
                                       {'id': 16,
                                        'name': 'vegetation',
                                        'supercategory': 'None',
                                        'isthing': 0,
                                        'ignore': False},
                                       {'id': 17,
                                        'name': 'building',
                                        'supercategory': 'None',
                                        'isthing': 0,
                                        'ignore': False},
                                       {'id': 18,
                                        'name': 'infrastructure',
                                        'supercategory': 'None',
                                        'isthing': 0,
                                        'ignore': False},
                                       {'id': 19,
                                        'name': 'fence',
                                        'supercategory': 'None',
                                        'isthing': 0,
                                        'ignore': False},
                                       {'id': 20,
                                        'name': 'billboard',
                                        'supercategory': 'None',
                                        'isthing': 0,
                                        'ignore': True},
                                       {'id': 21,
                                        'name': 'trafficsign',
                                        'supercategory': 'None',
                                        'isthing': 0,
                                        'ignore': False},
                                       {'id': 22,
                                        'name': 'mobilebarrier',
                                        'supercategory': 'None',
                                        'isthing': 0,
                                        'ignore': True},
                                       {'id': 23,
                                        'name': 'trash',
                                        'supercategory': 'None',
                                        'isthing': 0,
                                        'ignore': True}]}

            new_img_list = []
            for img in panoptic_json['images']:
                city, seq, frame_no, _, file_type = img['file_name'].split('_')
                _, suffix = os.path.splitext(img['file_name'])
                seq, frame_no = int(seq), int(frame_no)
                new_rgb_file = os.path.join(f'{city}', f'{city}_{seq:0>6}_{frame_no:0>6}{suffix}')
                new_img = img.copy()
                new_img['file_name'] = new_rgb_file
                new_img_list.append(new_img)

            new_json['images'] = new_img_list

            new_panoptic_dir = output_dir / ('panoptic_' + split)
            new_panoptic_dir.mkdir(parents=True, exist_ok=True)
            ann_id_counter = 1
            new_annotation_list = []
            for ann in tqdm(panoptic_json['annotations'], desc='Processing panoptic annotations'):
                old_fn = Path(ann['file_name'])
                city, seq, frame_no, _, file_type = old_fn.stem.split('_')
                seq, frame_no = int(seq), int(frame_no)
                new_fn = os.path.join(f'{city}', f'{city}_{seq:0>6}_{frame_no:0>6}.png')

                new_ann = {'image_id': f'{city}_{seq:0>6}_{frame_no:0>6}',
                           'file_name': new_fn}
                old_panoptic_mask = np.asarray(Image.open(cityscapes_panoptic_dir / old_fn))
                old_panoptic_mask = rgb2id(old_panoptic_mask)
                new_panoptic_mask = np.zeros_like(old_panoptic_mask)
                seg_info = []
                for seg in ann['segments_info']:
                    old_ann_id = seg['id']
                    new_seg = seg.copy()
                    new_cat_id = CityscapesToVIPERPanopticLabelMap[seg['category_id']]
                    if new_cat_id != 256:
                        new_seg['category_id'] = new_cat_id
                        new_seg['id'] = ann_id_counter
                        new_panoptic_mask[old_panoptic_mask == old_ann_id] = ann_id_counter
                        ann_id_counter += 1
                        seg_info.append(new_seg)
                new_panoptic_mask = id2rgb(new_panoptic_mask)

                new_file = new_panoptic_dir / new_fn
                new_file.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(new_panoptic_mask).save(new_file)

                new_ann['segments_info'] = seg_info

                new_annotation_list.append(new_ann)

            new_json['annotations'] = new_annotation_list

            panoptic_json_fn = output_dir / 'annotations' / ('panoptic_coco_format_' + split + '.json')
            panoptic_json_fn.parent.mkdir(parents=True, exist_ok=True)
            with open(panoptic_json_fn, 'w') as f:
                json.dump(new_json, f)
            print(f'Wrote panoptic JSON annotations to {panoptic_json_fn}')

            # Create instance annotations - standard COCO eval scripts need them even if we're not doing instance segmentation
            convert_panoptic_to_detection_coco_format(panoptic_json_fn,
                                                      new_panoptic_dir,
                                                      output_dir / 'annotations' / f'instances_coco_format_{split}.json',
                                                      True)


if __name__ == "__main__":
    dataset_dir = os.getenv("DETECTRON2_DATASETS", "datasets")
    dataset_dir = Path(dataset_dir)

    assert dataset_dir.is_dir(), f'Dataset {dataset_dir} does not exist! Check environment variable "DETECTRON2_DATASETS"'

    processCityscapes(dataset_dir)
    processVIPER(dataset_dir)
