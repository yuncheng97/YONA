python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 --use_env train.py \
    --gpu_id '0,1' \
    --data_path  /220019054/Dataset/CVC-VideoClinicDB \
    --save_path ./result \
    --model_name YONA \
    --lr 5e-4 \
    --backbone resnet50 \
    --epoch 50 \
    --train_clips 2 \
    --test_clips 2 \
    --scheduler step \
    --n_threads 6 \
    --batch_size 8 \


# python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 --use_env train.py  \
#         --data_path  /220019054/Dataset/LDPolypVideo \
#         --model_path /220019054/MICCAI23-YONA/model/yona_ld_2 \
#         --backbone resnet50 \
#         --epoch 20 \
#         --train_clips 2 \
#         --test_clips 2 \
#         --scheduler cos \
#         --n_threads 6 \
#         --batch_size 16 >/220019054/MICCAI23-YONA/log/yona_ld_2.log 2>&1 &