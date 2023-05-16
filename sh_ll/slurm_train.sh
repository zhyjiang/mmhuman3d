#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=8 --mem=40GB
#SBATCH --job-name="3D_96Base"
# we run on the gpu partition and we allocate 1 titanx gpu
##SBATCH -p gpu --gres=gpu:1 -x hendrixgpu05fl,hendrixgpu06fl,hendrixgpu07fl,hendrixgpu08fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu05fl
##SBATCH -p gpu --gres=gpu:1 -w hendrixgpu01fl,hendrixgpu02fl,hendrixgpu03fl,hendrixgpu04fl,hendrixgpu06fl
##SBATCH -p ml4good --gres=gpu:1 
#SBATCH -p ml4good --gres=gpu:1 

#SBATCH --time=7-25:00:00
#SBATCH --mail-type=END,FAIL#SBATCH --mail-user=lilei@di.ku.ku.
#your script, in this case: write the hostname and the ids of the chosen gpus.

#SBATCH --output=./zzslurm_ouput/slurm-depth_base_%j.out

echo $hostname
echo $CUDA_VISIBLE_DEVICES

# sh sh_ll/train.sh
# python tools/train.py configs/depth/depth_pretrained_h36m.py \
#     --work-dir work_dirs/depth_from_zy_slurm  --gpus 1 --resume-from work_dirs/zy_model/epoch_17.pth


# python tools/train.py configs/depth/depth_pretrained_h36m_SwimBase.py \
#     --work-dir work_dirs/depth_swin2_base_slurm  \
#      --resume-from work_dirs /depth_swin2_base/epoch_15.pth \
#     --gpus 1


# python tools/train.py configs/depth/depth_pretrained_h36m.py \
#     --work-dir work_dirs/depth_swin2_base  \
#      --resume-from work_dirs /depth_swin2_base/epoch_15.pth \
#     --gpus 1

# python tools/train.py configs/depth/depth_pretrained_h36m_SwimBase.py \
#     --work-dir work_dirs/depth_swin2_base_center96  \
#     --resume-from work_dirs/depth_swin2_base_center96/epoch_8.pth \
#     --gpus 1

## this is for input size 256 but the model is trained in 396
# python tools/train.py configs/depth/depth_pretrained_h36m_SwimBase256.py \
#     --work-dir work_dirs/depth_swin2_base  \
#     --resume-from work_dirs/depth_swin2_base/epoch_55.pth \
#     --gpus 1

export CUDA_VISIBLE_DEVICES=0

# python tools/train.py configs/depth/depth_pretrained_h36m_SwimBase_KP.py \
#     --work-dir work_dirs/depth_swin2_base_center96_KP  \
#     --gpus 1


python tools/test.py configs/depth/depth_pretrained_h36m_SwimBase.py \
    --work-dir work_dirs/depth_swin2_base_center96  \
    work_dirs/depth_swin2_base_center96/epoch_20.pth 
