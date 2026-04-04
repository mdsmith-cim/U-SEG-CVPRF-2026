from pathlib import Path
import os
import shutil
from tqdm.auto import tqdm

def main():
    """
    Deletes unused image sequence data for the Cityscapes dataset. Specifically:
     * All test and train images
     * All images in the future from the annotated samples.
    @return:
    """
    dataset_root = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    assert dataset_root.exists(), f'Dataset root folder {dataset_root} does not exist! Verify environment variable "DETECTRON2_DATASETS"'

    cityscapes_folder = dataset_root / "cityscapes"
    assert cityscapes_folder.exists(), f'Cityscapes folder {cityscapes_folder} does not exist!'
    img_folder = cityscapes_folder / 'leftImg8bit'
    seq_folder = cityscapes_folder / 'leftImg8bit_sequence'
    seq_folder_val = seq_folder / 'val'
    img_folder_val = img_folder / 'val'

    # Delete test and train folders in sequence if applicable as we only use sequence data at inference time
    print('Deleting test and train images...')
    shutil.rmtree(seq_folder / 'test', ignore_errors=True)
    shutil.rmtree(seq_folder / 'train', ignore_errors=True)

    annotated_img_list = list(img_folder_val.glob('**/*.png'))
    annotated_img_list.sort()

    # Delete all files in sequence that are in the future as we will not use them
    delete_counter = 0
    for img_path in tqdm(annotated_img_list, desc='Deleting future images for each annotated image'):
        city, seq, frame, remainder = img_path.name.split('_')
        rel_path_parent = img_path.relative_to(img_folder_val).parent
        # Cityscapes provides -19 / +10 images for each annotated one
        for i in range(1, 11):
            new_frame_num = f'{int(frame) + i:0>6}'
            new_fn = f'{city}_{seq}_{new_frame_num}_{remainder}'
            to_delete_fn = seq_folder_val / rel_path_parent / new_fn
            to_delete_fn.unlink(missing_ok=True)
            delete_counter += 1

    print(f'Deleted {delete_counter} future images')


if __name__ == '__main__':
    main()