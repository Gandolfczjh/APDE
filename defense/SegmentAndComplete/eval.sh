#!/bin/bash

# # 'results/tsea-pgd/tsea-pgd_faster_rcnn.png'
# # patch_path="/data/zjh/code/eval/results/tsea-pgd"
# advdataset_path="/data/zjh/dataset/adv_dataset/images"

# # conda init bash
# # conda activate pytorch

# # python --version

# for attack_path in ${advdataset_path}/*
# do 
#     for detector_i in ${attack_path}/*
#     do
#         python ./ex_SAC.py --advdataset_path $detector_i
#         echo "Finished python ./ex_SAC.py --advdataset_path $detector_i..."
#         sleep 2
#     done
# done

# advdataset_path="/data/zjh/dataset/size_exp/"

# attack_method=("large" "medium" "small")
# # device=$2

# for attack_method_i in "${attack_method[@]}"
# do
#     attack_path=$advdataset_path$attack_method_i

#     prepath_list=("/advpatch/retinanet" "/GNAP/yolov2" "/TCEGA/faster_rcnn" "/tsea/yolov5")

#     for prepath in "${prepath_list[@]}"
#     do
#         python ./ex_SAC.py --advdataset_path $attack_path$prepath
#         echo "Finished python ./ex_SAC.py --advdataset_path $attack_path$prepath..."
#         sleep 2
#     done
# done

# advdataset_path="/data/zjh/dataset/physical/images"

# position_list=("/OUT" "/INDOOR")
# distance_list=("/FAR" "/NEAR")
# angle_list=("/-45_-15" "/-15_15" "/15_45")

# for attack_method in ${advdataset_path}/*
# do
#     echo $attack_method
#     for position_i in "${position_list[@]}"
#     do
#         temp=$attack_method$position_i
#         for distance_i in "${distance_list[@]}"
#         do
#             temp1=$temp$distance_i
#             for angle_i in "${angle_list[@]}"
#             do
#                 temp2=$temp1$angle_i
#                 python ./ex_SAC.py --advdataset_path $temp2
#                 echo "Finished python ./ex_SAC.py --advdataset_path $temp2..."
#                 sleep 0.1
#             done
#         done
#     done
# done


advdataset_path="/data/zjh/dataset/adv_dataset/images"

attack_list=('advpatch' 'tsea' 'DM-NAP' 'GNAP' 'TCEGA')
detector_list=('yolov2' 'yolov5' 'yolov7' 'faster_rcnn')

for attack_i in "${attack_list[@]}"
do 
    for detector_i in "${detector_list[@]}"
    do
        temp=$advdataset_path/$attack_i/$detector_i
        python ./ex_SAC.py --advdataset_path $temp
        echo "Finished python ./ex_SAC.py --advdataset_path $temp..."
        sleep 0.1
    done
done