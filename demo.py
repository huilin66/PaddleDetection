import os.path
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import io
ROOT_PATH = r'/mnt/e/data/tp'
TAXI_PATH = os.path.join(ROOT_PATH, 'taxi_id.csv')
INTS_PATH = os.path.join(ROOT_PATH, 'intersections.csv')

def data_load():
    df_taxi = pd.read_csv(TAXI_PATH)
    df_taxi.columns = ['taxi_id', 'pack_up_time','drop_off_time', 'pack_up_intersection','drop_off_intersection',]
    print('records loaded!')


    df_taxi['start_date'] = pd.to_datetime(df_taxi['pack_up_time'], unit='s')
    df_taxi['end_date'] = pd.to_datetime(df_taxi['drop_off_time'], unit='s')
    df_taxi.set_index('start_date', inplace=True, drop=False)

    df_ints = pd.read_csv(INTS_PATH)
    df_ints.columns = ['id', 'latitude','longitude']
    print('intersection loaded!')
    return df_taxi, df_ints

def task1(df_taxi):
    taxi_ids = df_taxi['taxi_id'].unique()
    print('total taxis:', len(taxi_ids))
    print('total records:', len(df_taxi))

def task2(df_taxi):
    trip_counts = df_taxi['taxi_id'].value_counts().sort_values()
    x = trip_counts.index.to_list()
    y = list(trip_counts.values)

    plt.bar(np.arange(len(x)), y)
    plt.xticks(np.arange(len(x)), x)
    plt.title('the distribution of the number of trips')
    plt.xlabel('taxi id')
    plt.ylabel('number of trips')
    plt.savefig('task2.png')
    plt.close()

    # print(trip_counts)

    x_top = x[-10:]
    y_top = y[-10:]
    plt.bar(np.arange(len(x_top)), y_top)
    plt.xticks(np.arange(len(x_top)), x_top, rotation=30)
    plt.title('the distribution of the number of trips(top10)')
    plt.xlabel('taxi id')
    plt.ylabel('number of trips')
    plt.savefig('task2_top10.png')
    plt.close()
    print(x_top)
    print(y_top)

def task3(df_taxi):
    df_taxi.groupby(df_taxi.index.date).size().plot()
    plt.xlabel('date')
    plt.ylabel('number of trips')
    plt.title('the distribution of daily trip count')

    plt.savefig('task3.png')
    plt.close()

    date_range = pd.date_range(start='2011-01-01', end='2012-01-01')
    df = pd.DataFrame(date_range, columns=['date'])
    df['value'] = df['date'].apply(lambda x: 450000 if x.weekday() < 6 else 350000)
    plt.plot(df['date'], df['value'])

    df_taxi.groupby(df_taxi.index.date).size().plot()
    plt.xlabel('date')
    plt.ylabel('number of trips')
    plt.title('working days and rest days & daily trip count')
    plt.legend(['working days and rest days', 'real records'])

    plt.savefig('task3_dist.png')
    plt.close()

    df_taxi.groupby(df_taxi.index.weekday).size().plot(kind='bar')
    plt.xlabel('weekday')
    plt.xticks(np.arange(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], rotation=30)
    plt.ylabel('number of trips')
    plt.title('the distribution of trip count of the weekdays')
    plt.savefig('task3_week.png')
    plt.close()

def task4(df_taxi):
    trip_counts = df_taxi['pack_up_intersection'].value_counts().sort_values()
    plt.bar(trip_counts.index, trip_counts.values)
    plt.title('the distribution of departure trips')
    plt.xlabel('date')
    plt.ylabel('departure location')
    plt.savefig('task4_departure.png')
    plt.close()
    print(trip_counts)

    trip_counts = df_taxi['drop_off_intersection'].value_counts().sort_values()
    plt.bar(trip_counts.index, trip_counts.values)
    plt.title('the distribution of arrival trips')
    plt.xlabel('date')
    plt.ylabel('arrival location')
    plt.savefig('task4_arrival.png')
    plt.close()
    print(trip_counts)

def task5(df_taxi):
    df_taxi.groupby(df_taxi.index.hour).size().plot()
    plt.xlabel('hour')
    plt.ylabel('the distribution of the number of trips')
    plt.savefig('task5.png')
    plt.close()

def task6(df_taxi, df_ints):
    df_taxi['time'] = df_taxi['end_date'] - df_taxi['start_date']
    df_taxi['time'] = df_taxi['time'].dt.total_seconds()/60
    df_taxi['time'] = df_taxi['time'].astype(int)
    print(df_taxi['time'][:10])
    # df_taxi['time'].plot(kind='hist')
    plt.hist(df_taxi['time'], bins=20)
    plt.xlabel('travel time (minutes)')
    plt.ylabel('the probability distribution')
    plt.title('the probability distribution of travel time')
    plt.savefig('task6_time.png')
    plt.close()


    # df_taxi = df_taxi.merge(df_ints, left_on='pack_up_intersection', right_on='id', suffixes=('', '_s'))
    # df_taxi = df_taxi.merge(df_ints, left_on='drop_off_intersection', right_on='id', suffixes=('', '_e'))
    # df_taxi = df_taxi.drop(['id_s', 'id_e'], axis=1)
    # df_taxi['x_s'], df_taxi['y_s'], df_taxi['x_e'], df_taxi['y_e'] = map(np.radians,
    # [df_taxi['latitude_s'], df_taxi['longitude_s'], df_taxi['latitude_e'], df_taxi['longitude_e']])
    # df_taxi['delta_x'] = df_taxi['x_e'] - df_taxi['x_s']
    # df_taxi['delta_y'] = df_taxi['y_e'] - df_taxi['y_s']
    # a = np.sin(df_taxi['delta_x'] / 2) ** 2 + np.cos(df_taxi['x_s']) * np.cos(df_taxi['x_e']) * np.sin(df_taxi['delta_y'] / 2) ** 2
    # c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    # r = 6371
    # df_taxi['distance'] = c * r
    # df_taxi['distance'].plot(kind='hist')
    # plt.xlabel('travel distance (minutes)')
    # plt.ylabel('the probability distribution')
    # plt.savefig('task6_distance.png')
    # plt.close()
import cv2
from tqdm import tqdm
def imgs_clip(input_dir, output_dir):
    gapup = int(320 * 0.13)
    gapbt = int(320 * 0.22)
    gaplf = int(640 * 0.098)
    gaprg = int(640 * 0.13)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        img = io.imread(input_path)
        img = img[gapup:-gapbt, gaplf:-gaprg]
        io.imsave(output_path, img)

def imgs_rs(input_dir, output_dir, sta=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        img = io.imread(input_path)
        img = cv2.resize(img, dsize=(320, 240), interpolation=cv2.INTER_NEAREST)
        if sta:
            uniques = np.unique(img)
            if 5 in uniques or 6 in uniques:
                print(file_name)
                break
        io.imsave(output_path, img)


def cp_img(input_dir, ref_dir, output_dir):
    file_list = os.listdir(ref_dir)
    for file_name in tqdm(file_list):
        input_path = os.path.join(input_dir, file_name.replace('.png', '.jpg'))
        output_path = os.path.join(output_dir, file_name)
        shutil.copy(input_path, output_path)


def cmd_run(cmd_str):
    os.system(cmd_str)

if __name__ == '__main__':
    pass
    # df_taxi, df_ints = data_load()
    # task1(df_taxi)
    # task2(df_taxi)
    # task3(df_taxi)
    # task4(df_taxi)
    # task5(df_taxi)
    # task6(df_taxi, df_ints)
    # img = io.imread(r'/mnt/e/data/polyu/data202311/mt_img/FLIR4247.jpg')
    # gapup = int(320 * 0.13)
    # gapbt = int(320 * 0.22)
    # gaplf = int(640 * 0.098)
    # gaprg = int(640 * 0.13)
    # img = img[gapup:-gapbt, gaplf:-gaprg]
    # print(gapup,320-gapbt, gaplf,640-gaprg,)
    # print(320 - gapbt-gapup, 640 - gaprg-gaplf, )
    # plt.imshow(img)
    # # imgt = io.imread(r'/mnt/e/data/polyu/100_FLIR/FLIR3189.jpg')
    # # plt.imshow(imgt)
    # plt.show()

    # input_dir = r'/mnt/e/data/polyu/data202311seg/mt_img_segset/img'
    # output_dir = r'/mnt/e/data/polyu/data202311seg/mt_img_segset_clip/img'
    # imgs_clip(input_dir, output_dir)
    # input_dir = r'/mnt/e/data/polyu/data202311seg/mt_img_segset/gt'
    # output_dir = r'/mnt/e/data/polyu/data202311seg/mt_img_segset_clip/gt'
    # imgs_clip(input_dir, output_dir)
    # input_dir = r'/mnt/e/data/polyu/data202311seg/mt_img_segset/vis'
    # output_dir = r'/mnt/e/data/polyu/data202311seg/mt_img_segset_clip/vis'
    # imgs_clip(input_dir, output_dir)

    # input_dir = r'/mnt/e/data/polyu/data202311seg/mt_img_segset_clip/img'
    # output_dir = r'/mnt/e/data/polyu/data202311seg/mt_img_segset_clip_rs/img'
    # imgs_rs(input_dir, output_dir)
    # input_dir = r'/mnt/e/data/polyu/data202311seg/mt_img_segset_clip/gt'
    # output_dir = r'/mnt/e/data/polyu/data202311seg/mt_img_segset_clip_rs/gt'
    # imgs_rs(input_dir, output_dir, sta=True)
    # input_dir = r'/mnt/e/data/polyu/data202311seg/mt_img_segset_clip/vis'
    # output_dir = r'/mnt/e/data/polyu/data202311saeg/mt_img_segset_clip_rs/vis'
    # imgs_rs(input_dir, output_dir)

    # input_dir = r'/mnt/e/data/polyu/data202311/mt'
    # ref_dir = r'/mnt/e/data/polyu/data202311seg/mt_thermal_segset/gt'
    # output_dir = r'/mnt/e/data/polyu/data202311seg/mt_thermal_segset/img'
    # cp_img(input_dir, ref_dir, output_dir)
    # cmd_str_polyuz_infer_contrast_rs_ = r'python E:\repository\PaddleDetection\tools\infer.py ' \
    #                 r'-c E:\repository\PaddleDetection\configs\ppyoloe\ppyoloe_l_80e_defect.yml ' \
    #                 r'-o weights=E:\repository\PaddleDetection\output\best_model.pdparams ' \
    #                 r'--infer_img=E:\repository\defectdet\BDDetection\dataddd\input\1701661721.7274697.jpg ' \
    #                 r'--output_dir=E:\repository\defectdet\BDDetection\dataddd\vis --slice_infer --slice_size 640 640 --overlap_ratio 0.1 0.1'
    # cmd_str_ppyoloe_eval = r'python E:\repository\PaddleDetection\tools\eval.py ' \
    #                        r'-c E:\repository\PaddleDetection\configs\ppyoloe\ppyoloe_l_80e_defect.yml ' \
    #                        r'-o weights=E:\repository\PaddleDetection\output\best_model.pdparams ' \
    #                        r'--classwise'
    cmd_str_ppyoloe_train = r'python E:\repository\PaddleDetection\tools\train.py ' \
                           r'-c E:\repository\PaddleDetection\configs\ppyoloe\ppyoloe_crn_l_300e_coco_wt.yml ' \
                           r'--eval'
    # cmd_str_ppyoloe_train = r'python E:\repository\PaddleDetection\tools\train.py ' \
    #                        r'-c E:\repository\PaddleDetection\configs\ppyoloe\ppyoloe_l_80e_defect.yml ' \
    #                        r'--eval'
    cmd_run(cmd_str_ppyoloe_train)





