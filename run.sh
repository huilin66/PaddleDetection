#python data_convert.py \
#        --anno-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset \
#        --save-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset_coco \
#        --image-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset \
#        --json-name /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset_coco.json

#python data_convert.py \
#        --anno-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/VALID/O6 \
#        --save-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/VALID/valset_coco \
#        --image-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/VALID/O6 \
#        --json-name /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/VALID/valset_coco.json

#python data_convert.py \
#        --anno-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TEST/test \
#        --save-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TEST/test_coco \
#        --image-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TEST/test \
#        --json-name /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TEST/test_coco.json


#python data_convert.py \
#        --anno-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset_aos \
#        --save-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset_aos_coco \
#        --image-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset_aos \
#        --json-name /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset_aos_coco.json
#
#python data_convert.py \
#        --anno-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/VALID/valset_aos \
#        --save-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/VALID/valset_aos_coco \
#        --image-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/VALID/valset_aos \
#        --json-name /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/VALID/valset_aos_coco.json

python data_convert.py \
        --anno-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TEST/testset_aos \
        --save-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TEST/testset_aos_coco \
        --image-path /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TEST/testset_aos \
        --json-name /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TEST/testset_aos_coco.json




#python data_split.py \
#        --src_dir /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset \
#        --aos_dir /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset_aos \
#        --ahe_dir /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TRAIN/trainset_ahe


#python data_split.py \
#        --src_dir /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/VALID/O6 \
#        --aos_dir /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/VALID/valset_aos \
#        --ahe_dir /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/VALID/valset_ahe

#python data_split.py \
#        --src_dir /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TEST/test \
#        --aos_dir /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TEST/testset_aos \
#        --ahe_dir /mnt/e/data/Search_and_Rescue_with_Airborne_Optical_Sectioning/YOLO/YOLO/data/SARAOS/img/TEST/testset_ahe
