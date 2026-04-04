import argparse
import contextlib
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from joblib import Parallel, delayed, parallel_backend
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import torchvision.transforms.functional as F
from tqdm.auto import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# Based on https://stackoverflow.com/questions/33564246/passing-a-tuple-as-command-line-argument
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def get_args_parser():
    parser = argparse.ArgumentParser('Optical flow calculator for cityscapes')
    parser.add_argument('--set', type=str, help='Image set to use.', default='val', choices=['train', 'val'])
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for RAFT optical flow generation')
    parser.add_argument('--num_jobs', type=int, default=len(os.sched_getaffinity(0)),
                        help="Number of jobs to run simultaneously")
    parser.add_argument('--verbose', action='store_true', help='Show debug output')
    # 3/4 normal size for Cityscapes (1536, 768) seems to fit in VRAM. Normal size (2048,1024) does not
    parser.add_argument('--resize', default=None, type=tuple_type, help='Size to resize image to. Format: (H, W)')
    return parser

def loadImages(img_folder, cur_image_filename, prev_image_filename, transforms, size, device):
    """
    Loads images from disk and applies necessary transforms to map to [-1, 1] in float32
    :param img_folder: Path to image folder
    :param cur_image_filename: Filename of current (second) image
    :param prev_image_filename: Filename of previous (first) image
    :param device: Device to store image data on
    :return: tuple (img1 batch, img2 batch)
    """
    img1, img2 = transforms(Image.open(img_folder / prev_image_filename), Image.open(img_folder / cur_image_filename))
    original_size = img1.shape[-2:]
    assert original_size == img2.shape[-2:], f'Image sizes of {cur_image_filename} and {prev_image_filename} expected to be consistent!'
    if size is not None:
        # Apparently no antialiasing was used during training
        img1 = F.resize(img1, size=size, antialias=False)
        img2 = F.resize(img2, size=size, antialias=False)
    # Stack up img1 and img2 such that we can calculate forward and backward flow at the same time
    img1_batch = torch.stack([img1, img2]).to(device)
    img2_batch = torch.stack([img2, img1]).to(device)
    return img1_batch, img2_batch, original_size


@torch.no_grad()
def singleThreadOptFlow(img_folder, opt_flow_folder_fw, opt_flow_folder_bw, model, transforms, size, device, prev_image, cur_image):

    city_folder = cur_image.parent
    assert city_folder == prev_image.parent, 'Current and previous images must be in the same sequence folder!'
    cur_filename_stem = cur_image.stem
    prev_filename_stem = prev_image.stem

    img1_batch, img2_batch, original_size = loadImages(img_folder, cur_image, prev_image, transforms, size, device)

    # Verify size is divisible by 8
    h, w = img1_batch.shape[-2:]
    assert (h % 8 == 0) and (w % 8 == 0), f'Image sizes must be divisible by 8! Found size ({h}, {w})'

    # First dim is model iteration number; we use only last iteration
    predicted_flow = model(img1_batch, img2_batch)[-1]

    # Predicted flow has shape (2 [forward/backward flow], 2 [u/v], h, w)
    predicted_flow = predicted_flow.cpu().to(torch.float16).numpy()

    dest_folder_fw = opt_flow_folder_fw / city_folder
    dest_folder_bw = opt_flow_folder_bw / city_folder
    dest_folder_fw.mkdir(parents=True, exist_ok=True)
    dest_folder_bw.mkdir(parents=True, exist_ok=True)

    # Saving as float16, compressed, and as two separate variables is significantly more space efficient than
    # * Uncompressed array
    # * Compressed array as one variable
    # * Anything as float32
    # Here we save both forward and backward flow
    np.savez_compressed(dest_folder_fw / prev_filename_stem, u=predicted_flow[0, 0, ...],
                        v=predicted_flow[0, 1, ...])
    np.savez_compressed(dest_folder_bw / cur_filename_stem, u=predicted_flow[1, 0, ...],
                        v=predicted_flow[1, 1, ...])


@torch.no_grad()
def main(args):

    image_set = args.set
    verbose = args.verbose

    dataset_root = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    assert dataset_root.exists(), f'Dataset root folder {dataset_root} does not exist!'

    dataset_folder = dataset_root / 'cityscapes'
    assert dataset_folder.is_dir(), f'Cityscapes dataset folder {dataset_folder} does not exist!'
    seq_folder = dataset_folder / 'leftImg8bit_sequence' / image_set
    seq_img_list = list(seq_folder.glob('**/*.png'))
    opt_flow_folder_fw = dataset_folder / 'optflow_fw' / image_set
    opt_flow_folder_bw = dataset_folder / 'optflow_bw' / image_set
    opt_flow_folder_fw.mkdir(parents=True, exist_ok=True)
    opt_flow_folder_bw.mkdir(parents=True, exist_ok=True)

    seq_img_list.sort()

    print(f'Found {len(seq_img_list)} images in {seq_folder}')

    image_pairs = []

    prev_city = None
    prev_sequence = None
    prev_frameID = -1
    prev_image = None

    # Generate list of 2-image pairs to process flow for
    for i in seq_img_list:

        cur_image = i
        cur_city, cur_seq, cur_frame, remainder = cur_image.name.split('_')
        assert remainder == 'leftImg8bit.png', f'Expected leftImg8bit.png but got {remainder} for img {cur_image}'

        # New city
        new_city = cur_city != prev_city
        new_seq = prev_sequence != cur_seq
        if verbose and new_city:
            print(f'Transition from city {prev_city} to city {cur_city}')
        if verbose and new_seq:
            print(f'Transition from sequence {prev_sequence} to {cur_seq}')

        if new_seq or new_city:
            prev_city = cur_city
            prev_sequence = cur_seq
            prev_frameID = cur_frame
            prev_image = cur_image
            continue

        else:
            # Still current sequence
            if int(cur_frame) == (int(prev_frameID) + 1):
                prev_frameID = cur_frame
            # Some images (frankfurt) are sampled from the same sequence but at a much later time
            # This is effectively a new sequence
            elif int(cur_frame) > int(prev_frameID):
                if verbose:
                    print(f'Sequence gap! Between {cur_image} and {prev_image}')
                prev_frameID = cur_frame
                prev_image = cur_image
                continue
            else:
                raise Exception(f'Sequence mismatch! Between {cur_image} and {prev_image}')

            if prev_image is not None:
                image_pairs.append({'prev_image': prev_image.relative_to(seq_folder), 'cur_image': cur_image.relative_to(seq_folder)})
                prev_image = cur_image
            else:
                print(f'Skipping {cur_image}')
                continue

    print(f'Generated {len(image_pairs)} image pairs')

    # Sanity checking
    for pair in image_pairs:
        prev_image = pair['prev_image']
        cur_image = pair['cur_image']
        prev_city, prev_seq, prev_frame, prev_remainder = prev_image.name.split('_')
        cur_city, cur_seq, cur_frame, cur_remainder = cur_image.name.split('_')
        prev_city_folder = prev_image.parent.name
        cur_city_folder = cur_image.parent.name
        assert prev_city == cur_city, 'Cities must be the same!'
        assert prev_seq == cur_seq, 'Sequences must be the same!'
        assert (int(prev_frame) + 1) == int(cur_frame), f'Frames must be increasing! {pair}'
        assert prev_remainder == cur_remainder, 'Remainder must be the same!'
        assert prev_city_folder == cur_city_folder, 'City folders must be the same!'


    # Use default weights; they work best in general. KITTI weights don't even work that well on KITTI!
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(args.device)
    model = model.eval()
    transforms = weights.transforms()

    if args.resize is not None:
        first_img_size = Image.open(seq_img_list[0]).size
        print(f'Resizing images to {args.resize}')
        print(f'First image original size: {first_img_size}')

    # Run multithreaded
    # This is not the best way for pytorch (originally written for CPU) but it keeps the GPU busy enough and is fast
    # so not worth my time to improve. Can also be run on CPU only where it works well at consuming it entirely
    max_cpus_avail = len(os.sched_getaffinity(0))
    n_jobs = args.num_jobs
    thread_per_job = 1
    print(f'{max_cpus_avail} CPU cores available')
    print(f'Running {n_jobs} jobs with {thread_per_job} threads per job')
    with tqdm_joblib(tqdm(desc='Optical Flow', total=len(image_pairs))) as progress_bar:
        with parallel_backend('loky', inner_max_num_threads=thread_per_job):
            Parallel(n_jobs=n_jobs)(
                delayed(singleThreadOptFlow)(seq_folder, opt_flow_folder_fw, opt_flow_folder_bw, model, transforms, args.resize, args.device,
                                             **i) for i in image_pairs)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
