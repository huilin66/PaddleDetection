import os
import pandas as pd
from tqdm import tqdm
import numpy as np


def name_img2gt(img_name):
    gt_name = img_name.replace('.jpg','.png')
    return gt_name
def name_img2f(img_name):
    f_name = img_name.replace('.jpg','.tif')
    return f_name

def dataset_write(dataset_dict, csv_path):
    '''
    写入数据集split.csv
    :param dataset_dict: 数据集字典
    :param csv_path: csv路径
    :return:
    '''
    df = pd.DataFrame(dataset_dict,)
    df.to_csv(csv_path, header=None, index=None, sep=' ')

def dir2txt(input_dir, dst_path, split_rate=1.0):
    img_dir = os.path.join(input_dir, 'images')
    mask_dir = os.path.join(input_dir, 'labels')

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    files_list = os.listdir(img_dir)
    img_list = []
    mask_list = []
    for file_name in tqdm(files_list):
        img_path = os.path.join(img_dir, file_name)
        mask_path = os.path.join(mask_dir, name_img2gt(file_name))
        img_list.append(img_path)
        mask_list.append(mask_path)

    info_dict = {'img_path': img_list,
                'gt_path': mask_list,
                }
    dataset_write(info_dict, dst_path)

def imgnames2txt(imgs_dir, txt_path, random_seed=322):
    print('dataset sta:')
    files_list = os.listdir(imgs_dir)

    # region 普通的随机打乱，val样本可能会与train样本有部分图像块上的重叠
    np.random.seed(random_seed)
    np.random.shuffle(files_list)


    files_list = [os.path.join(imgs_dir, file_name) for file_name in files_list]

    info_dict = {'img_path1': files_list,}

    dataset_write(info_dict, txt_path)
    print('%d img finished\n'%len(files_list))

def dataset_split(imgs_path1, save_path, gts_path=None, train_rate=0.9):
    '''
    根据img文件列表，随机划分为不同数据集，并写入csv中
    :param imgs_path1: img1路径
    :param imgs_path2: img2路径
    :param gts_path: gt路径
    :param save_path: csv保存路径
    :param train_rate: 训练集比例
    :return:
    '''
    print('dataset split:')
    files_list = os.listdir(imgs_path1)

    # region 普通的随机打乱，val样本可能会与train样本有部分图像块上的重叠
    np.random.seed(322)
    np.random.shuffle(files_list)
    train_num = int(train_rate * len(files_list))
    val_num = len(files_list) - train_num
    files_list_train = files_list[:train_num]
    files_list_val = files_list[train_num:]
    # endregion

    # img_train_list1 = [os.path.join(os.path.basename(imgs_path1), file_name) for file_name in files_list_train]
    # img_val_list1 = [os.path.join(os.path.basename(imgs_path1), file_name) for file_name in files_list_val]

    img_train_list1 = files_list_train
    img_val_list1 = files_list_val

    if gts_path is not None:
        gt_train_list = [os.path.join(os.path.basename(gts_path), name_img2gt(file_name)) for file_name in files_list_train]
        gt_val_list = [os.path.join(os.path.basename(gts_path), name_img2gt(file_name)) for file_name in files_list_val]

        info_dict = {'img_path1': img_train_list1,
                'gt_path': gt_train_list,
                }
        dataset_write(info_dict, save_path)

        info_dict = {'img_path1': img_val_list1,
        'gt_path': gt_val_list,
        }
        dataset_write(info_dict, save_path.replace('train.txt','val.txt'))
        print('train rate %.2f\ntrain %d, val %d\nresult save to %s' % (train_rate, train_num, val_num, save_path))
        print('finish\n')
    else:
        info_dict = {'img_path1': img_train_list1,
                }
        dataset_write(info_dict, save_path)

        info_dict = {'img_path1': img_val_list1,
                }
        dataset_write(info_dict, save_path.replace('train.txt','val.txt'))
        print('train rate %.2f\ntrain %d, val %d\nresult save to %s' % (train_rate, train_num, val_num, save_path))
        print('finish\n')

if __name__ == '__main__':
    # imgs_dir = r'/mnt/e/data/polyu/ExpData/PolyUOutdoor_UAV'
    imgs_dir = r'/mnt/e/data/polyu/ExpData/PolyUOutdoor_UAV/images'
    txt_path = r'/mnt/e/data/polyu/ExpData/PolyUOutdoor_UAV/train.txt'
    # dir2txt(imgs_dir, txt_path)
    dataset_split(imgs_dir, txt_path, train_rate=1)