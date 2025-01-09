# OMP_NUM_THREADS=12 python -W ignore -m torch.distributed.run --nnodes=1 --nproc_per_node=8 main_dino.py \
# --arch swin_tiny \
# --data_path ./luperson \
# --output_dir ./log/lup/dino_tiny \
# --height 256 --width 128 \
# --crop_height 128 --crop_width 64 \
# --epochs 50 \
# --batch_size_per_gpu 24 \
# --global_crops_scale 0.8 1. \
# --local_crops_scale 0.05 0.8 \

python main_dino.py \
--arch swin_tiny \
--data_path ./luperson \
--output_dir ./log/lup/dino_tiny \
--height 224 --width 224 \
--crop_height 112 --crop_width 112 \
--epochs 100 \
--batch_size_per_gpu 12 \
--global_crops_scale 0.8 1. \
--local_crops_scale 0.05 0.8 \
--use_fp16 true \
--num_workers 4
