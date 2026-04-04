import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import wait
from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import Value
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from detectron2.data import MetadataCatalog
from detectron2.data.build import get_detection_dataset_dicts
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# For importing metadata registry
from mask2former import data  # noqa
from offline_eval.metrics import aurc, eaurc
from offline_eval.uncert import getConfidence
from offline_eval.utils import default_setup, loadConfig

DATASET_ROOT = Path(os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets")))
assert DATASET_ROOT.exists(), f'Dataset directory does not exist: {DATASET_ROOT}'

tqdm_position = Value('i', 0)


def arg_parser():
    """
    Create a parser with options for config file spec as well as config overrides.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Offline Evaluation")
    parser.add_argument('experiment_directory', help='Directory in which to search for model output for processing',
                        type=Path)
    parser.add_argument('--delete', action='store_true',
                        help='Delete .hdf5 files from disk when complete to save space.')
    parser.add_argument('--num_workers', default=None, type=int,
                        help='Number of parallel threads to use for processing results.')
    parser.add_argument('--yes', action='store_true', help='Assume yes to delete prompt')
    return parser


def find_folders(experiment_directory: Path) -> list[Path]:
    logger = logging.getLogger("offline_preprocess.find_folders")

    assert experiment_directory.is_dir(), f'{experiment_directory} must be a directory!'
    available_folders = list(experiment_directory.glob('*/'))

    logger.info(f'Detected {len(available_folders)} folders')

    return available_folders


def main(folder: Path):
    logger = logging.getLogger("offline_preprocess.main")

    rel_folder = folder.stem
    logger.info(f'Processing {rel_folder}...')

    inf_folder = folder / 'inference'
    assert inf_folder.is_dir(), f'Expect folder `inference` to exist in {folder}!'

    # Check existence of /integrity / ability to load required files
    h5_filename = inf_folder / 'model_output.hdf5'

    cfg = loadConfig(folder / 'config.yaml')
    h5file = h5py.File(h5_filename, "r")

    backbone = cfg['MODEL']['BACKBONE']['NAME']
    if backbone == 'build_resnet_backbone':
        depth = cfg['MODEL']['RESNETS']['DEPTH']
        backbone = 'ResNet' + str(depth)
    elif backbone == 'D2SwinTransformer':
        embed_dim = cfg['MODEL']['SWIN']['EMBED_DIM']
        assert embed_dim == 128, 'Expect SWIN model to be base with embed dim 128'
        backbone = 'Swin-B'
    elif backbone == 'D2TimmModel':
        backbone = cfg['MODEL']['TIMMMODEL']['MODEL_NAME']

    dataset = cfg["DATASETS"]["TEST"]
    assert len(dataset) == 1, 'Expect only 1 dataset!'
    dataset = dataset[0]

    entry = {'Folder': rel_folder, 'Backbone': backbone, 'Dataset': dataset}

    prediction_model = []

    uncert_entry = cfg['MODEL']['MASK_FORMER']['UNCERTAINTY']

    if uncert_entry['MC_DROPOUT_ENABLED']:
        prediction_model.append('MCDropout')
        entry["MC Samples"] = uncert_entry['MC_DROPOUT_SAMPLES']
    if uncert_entry['TTA_ENABLED']:
        prediction_model.append('TTA')
        entry["TTA Transforms"] = uncert_entry['TTA_TRANSFORMS']
    if uncert_entry['TIMESERIES_NUM_PREV_FRAMES'] > 0:
        prediction_model.append('Timeseries')
        entry["# of Frames"] = uncert_entry['TIMESERIES_NUM_PREV_FRAMES']
    if len(prediction_model) == 0:
        prediction_model = 'Softmax'
    else:
        prediction_model = "+".join(prediction_model)

    pan_aggregation_method = uncert_entry['PANOPTIC_AGGREGATION_METHOD']
    if pan_aggregation_method == 'averaging':
        entry['Aggregation Overlap Threshold'] = uncert_entry['AGGREGATION_OVERLAP_THRESHOLD']
        entry['Aggregation Vote Threshold'] = uncert_entry['AGGREGATION_VOTE_THRESHOLD']

    entry['Panoptic Aggregation Method'] = pan_aggregation_method
    entry['Semantic Aggregation Method'] = uncert_entry['SEMANTIC_AGGREGATION_METHOD']
    entry.update({'Prediction Model': prediction_model, 'Seed': cfg["SEED"]})

    metadata = MetadataCatalog.get(dataset)

    panoptic_folder = Path(metadata.panoptic_root)
    panoptic_json = Path(metadata.panoptic_json)

    assert panoptic_folder.exists(), f'Dataset panoptic folder {panoptic_folder} does not exist!'

    with open(panoptic_json, "r") as f:
        panoptic_json_data = json.load(f)
    img_info_lookup = {i['id']: i for i in panoptic_json_data['images']}

    dataset_catalog_entry = get_detection_dataset_dicts(dataset, filter_empty=False)

    with tqdm_position.get_lock():
        cur_tqdm_pos = tqdm_position.value
        tqdm_position.value += 1

    ood_mode = dataset in ('cityscapes_fine_panoptic_ood_timeseries_val', 'cityscapes_fine_panoptic_ood_val',
                           'cityscapes_VIPER_format_val_panoptic_with_semseg_time_with_ood',
                           'cityscapes_VIPER_format_val_panoptic_with_semseg_with_ood')

    if ood_mode:

        per_image_data = []
        for e in tqdm(dataset_catalog_entry, position=cur_tqdm_pos, desc='Loading uncertainty data...'):
            rel_fn = Path(e['file_name']).relative_to(DATASET_ROOT)
            h5_fn = str(rel_fn.with_suffix(''))
            h5data = h5file[h5_fn]
            confidence = getConfidence(h5data, mode='ood')

            per_image_data.append(
                {'fn': h5_fn, 'confidence': confidence, 'is_ood': e.get('is_ood', False), 'image_id': e['image_id']})

        assert len(per_image_data) % 2 == 0, 'Expect 2 of each image'

        # Calculate OOD detection rate for each uncertainty type and aggregation level
        # Assumes dataset is in order (original image, ood image, original, ood, ...)
        ood_det_rate_results = defaultdict(int)
        for i in range(1, len(per_image_data), 2):
            cur_img = per_image_data[i]
            prev_img = per_image_data[i - 1]

            assert cur_img['image_id'] == prev_img[
                'image_id'], 'Image ID mismatch! Expect sequence of original and OOD images'
            assert cur_img['is_ood'], 'Expect to see current image as OOD'
            assert not prev_img['is_ood'], 'Expect previous image to not be OOD'
            for cur_conf, prev_conf in zip(cur_img['confidence'].items(), prev_img['confidence'].items()):
                uncert_type = cur_conf[0]
                assert uncert_type == prev_conf[0], 'Uncertainty type mismatch'
                for (cur_agg_type, cur_agg_val), (prev_agg_type, prev_agg_val) in zip(cur_conf[1].items(),
                                                                                      prev_conf[1].items()):
                    assert cur_agg_type == prev_agg_type, 'Uncertainty type mismatch'
                    # Uncertainty: expect greater uncertainty/confidence in OODness when we see ood
                    if cur_agg_val > prev_agg_val:
                        ood_det_rate_results[(uncert_type, cur_agg_type)] += 1

        # Convert count into an OOD detection rate
        # Of all images evaluated, what portion was the uncertainty higher/confidence lower for the image with OOD
        # corruption
        total_ood_samples = len(per_image_data) / 2
        for i, j in ood_det_rate_results.items():
            ood_det_rate_results[i] = j / total_ood_samples

        # Start collecting data for ROC curve
        # Get y_true and y_score
        roc_data = defaultdict(list)
        for i in per_image_data:
            conf = i['confidence']
            is_ood = i['is_ood']
            for unc_type, unc_data in conf.items():
                for agg_type, agg_data in unc_data.items():
                    roc_data[(unc_type, agg_type)].append((is_ood, agg_data))  # (label, score)

        ood_results = []
        for unc_type_d, agg_type_d in ood_det_rate_results.keys():

            y_true, y_score = [], []
            for yt, ys in roc_data[(unc_type_d, agg_type_d)]:
                y_true.append(yt)
                y_score.append(ys)

            y_true = np.array(y_true, dtype=int)
            y_score = np.array(y_score)

            auc_score = roc_auc_score(y_true, y_score)

            entry_copy = entry.copy()

            entry_copy['Uncertainty Measure'] = unc_type_d
            entry_copy['Aggregation Type'] = agg_type_d
            entry_copy['OOD Detection Rate'] = ood_det_rate_results[(unc_type_d, agg_type_d)]
            entry_copy['OOD AUROC'] = auc_score
            ood_results.append(entry_copy)

        with open(folder / 'ood_processed_results.json', 'w') as f:
            json.dump(ood_results, f)

        df = pd.DataFrame(ood_results)
        df_str = df.to_string(
            columns=['Folder', 'Seed', 'OOD Detection Rate', 'OOD AUROC', 'Uncertainty Measure', 'Aggregation Type'])
        logger.info(f'For {folder}:\n{df_str}')

    # Normal non-OOD analysis using PQ/DICE score for segmentation
    else:

        pan_file = inf_folder / 'raw_pq_data.pkl'
        with open(pan_file, "rb") as f:
            # Not exactly the safest way to store data
            panoptic_data = pickle.load(f)

        pan_file_summary = inf_folder / "panoptic_results.json"
        with open(pan_file_summary, "r") as f:
            panoptic_data_summary = json.load(f)

        sem_file = inf_folder / 'sem_seg_evaluation_v2.pth'
        sem_data = torch.load(sem_file, weights_only=False)

        # Keep track of semantic / panoptic performance
        for name in ['All', 'Global', 'Things', 'Stuff']:
            entry['PQ-' + name] = panoptic_data_summary[name]['pq']
            entry['RQ-' + name] = panoptic_data_summary[name]['rq']
            entry['SQ-' + name] = panoptic_data_summary[name]['sq']

        entry['mIoU'] = sem_data['mIoU'] / 100
        entry['mDICE'] = sem_data['mDICE'] / 100
        entry['gIoU'] = sem_data['gIoU'] / 100
        entry['gDICE'] = sem_data['gDICE'] / 100

        per_img_data = {'risk_pq': [], 'risk_iou': [], 'risk_dice': [], 'confidence': []}

        # Bit of a hackjob to deal with cityscapes path weirdness
        # Ideally this would be fixed in the panoptic JSON data but I'm not diving into that mess; good chunk is from cityscapesscripts itself
        is_cityscapes_dataset = ('cityscapes' in metadata.name) and ('VIPER_format' not in metadata.name)

        # Ideally would make hdf5 reading parallel but this requires MPI and is rather complex to rework
        # For future implementations might be best to use something other than HDF5
        for img_id, v in tqdm(panoptic_data.items(), desc=f'Calculating metrics for folder {rel_folder}',
                              position=cur_tqdm_pos):
            img_metrics = v.pq_average()
            pq = img_metrics['pq']
            img_info = img_info_lookup[img_id]
            img_fn = img_info['file_name']
            if is_cityscapes_dataset:
                city = img_fn.split('_')[0]
                img_fn = os.path.join(city, img_fn.replace('_gtFine', ''))
            prefix = Path(metadata.image_root).relative_to(DATASET_ROOT)
            img_fn_full = prefix / img_fn
            img_fn_full = str(img_fn_full.with_suffix(''))
            h5data = h5file[img_fn_full]

            assert img_id == img_info['id'] == h5data.attrs['image_id'], 'Image ID mismatch!'
            sem_v = sem_data['perImage'][img_fn_full]
            # DICE and IoU stored as 0-100
            # PQ is 0-1
            iou = sem_v['IoU'] / 100
            dice = sem_v['DICE'] / 100
            risk_pq = 1 - pq
            risk_iou = 1 - iou
            risk_dice = 1 - dice

            confidence = getConfidence(h5data, mode='aurc')
            assert len(confidence) > 0, 'Expect a non-empty confidence!'

            per_img_data['risk_pq'].append(risk_pq)
            per_img_data['risk_iou'].append(risk_iou)
            per_img_data['risk_dice'].append(risk_dice)
            per_img_data['confidence'].append(confidence)

        uncert_types = list(per_img_data['confidence'][0].keys())
        aggregation_types = list(per_img_data['confidence'][0][uncert_types[0]].keys())

        risk_pq = np.array(per_img_data['risk_pq'])
        risk_iou = np.array(per_img_data['risk_iou'])
        risk_dice = np.array(per_img_data['risk_dice'])

        results = []
        for u in uncert_types:
            for a in aggregation_types:
                selected_conf = np.array([c[u][a] for c in per_img_data['confidence']])

                aurc_pq = aurc(risk_pq, selected_conf)
                eaurc_pq = eaurc(risk_pq, selected_conf)

                aurc_iou = aurc(risk_iou, selected_conf)
                eaurc_iou = eaurc(risk_iou, selected_conf)

                aurc_dice = aurc(risk_dice, selected_conf)
                eaurc_dice = eaurc(risk_dice, selected_conf)

                entry_copy = entry.copy()

                entry_copy['Uncertainty Measure'] = u
                entry_copy['Aggregation Type'] = a
                entry_copy['AURC-PQ'] = aurc_pq
                entry_copy['EAURC-PQ'] = eaurc_pq
                entry_copy['AURC-IoU'] = aurc_iou
                entry_copy['EAURC-IoU'] = eaurc_iou
                entry_copy['AURC-DICE'] = aurc_dice
                entry_copy['EAURC-DICE'] = eaurc_dice
                results.append(entry_copy)

        with open(folder / 'processed_results.json', 'w') as f:
            json.dump(results, f)

        df = pd.DataFrame(results)
        df_str = df.to_string(
            columns=['Folder', 'Seed', 'PQ-Global', 'AURC-PQ', 'EAURC-PQ', 'gDICE', 'AURC-DICE', 'EAURC-DICE',
                     'Uncertainty Measure', 'Aggregation Type'])
        logger.info(f'For {folder}:\n{df_str}')

    return h5_filename


if __name__ == "__main__":
    args = arg_parser().parse_args()
    logger = default_setup(args, "offline_preprocess")
    folders = find_folders(args.experiment_directory)
    max_workers = args.num_workers if args.num_workers is not None else len(os.sched_getaffinity(0))

    exit_code = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_jobs = {executor.submit(main, folder=folder): folder for folder in folders}
        jobs = wait(future_jobs)
        num_not_finished = len(jobs.not_done)
        if num_not_finished > 0:
            logger.critical(f'{num_not_finished} jobs not finished!')
            for future in jobs.not_done:
                selected_folder = future_jobs[future]
                logger.critical(f'Folder {selected_folder} not finished!')
                exit_code = 1
        for future in jobs.done:
            selected_folder = future_jobs[future]
            try:
                h5_filename = future.result()
                # Really should not be any ways this would trigger without causing an exception in the above line first
                if not isinstance(h5_filename, Path):
                    logger.critical(f'Folder {selected_folder} returned non-Path object!')
                    exit_code = 1
                    continue
                if args.delete:
                    if args.yes:
                        choice = True
                    else:
                        choice = input(f'Delete file {h5_filename}? (Y/N)')
                        choice = choice in ('y', 'Y')
                    if choice and h5_filename.is_file():
                        h5_filename.unlink()
                        logger.info(f'Deleted file {h5_filename}')
                    else:
                        logger.info('Skipping HDF5 deletion.')

            except Exception as exc:
                logger.critical(f'Exception when processing {selected_folder}: {exc}')
                exit_code = 1

            else:
                logger.info(f'Processed {selected_folder}.')

    sys.exit(exit_code)