import os
import shutil
from tqdm import tqdm
import argparse

src_dir = r'/mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset'
aos_dir = r'/mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset_aos'
ahe_dir = r'/mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset_ahe'

def data_split(input_dir, output_dir1, output_dir2):
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)
    for file_name in tqdm(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, file_name)
        if 'conAHE' in file_name:
            output_path = os.path.join(output_dir2, file_name)
        else:
            output_path = os.path.join(output_dir1, file_name)
        # print(input_path)
        # print(output_path)
        shutil.copy(input_path, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', default='./data/images')
    parser.add_argument('--aos_dir', default='train.json')
    parser.add_argument('--ahe_dir', default='train.json')

    args = parser.parse_args()
    data_split(args.src_dir, args.aos_dir, args.ahe_dir)