#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=8 --mem=40GB
#SBATCH --job-name="Dist"
# we run on the gpu partition and we allocate 1 titanx gpu
##SBATCH -p gpu --gres=gpu:1 -x hendrixgpu05fl,hendrixgpu06fl,hendrixgpu07fl,hendrixgpu08fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu05fl
##SBATCH -p gpu --gres=gpu:1 -w hendrixgpu01fl,hendrixgpu02fl,hendrixgpu03fl,hendrixgpu04fl,hendrixgpu06fl
##SBATCH -p ml4good --gres=gpu:1 
#SBATCH -p ml4good --gres=gpu:2


#SBATCH --output=./slurm_ouput/slurm-depth_swin_tiny_%j.out

#SBATCH --time=6-25:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lilei@di.ku.du.
#your script, in this case: write the hostname and the ids of the chosen gpus.


echo $hostname
echo $CUDA_VISIBLE_DEVICES

#SBATCH --output=./slurm_ouput/slurm-depth_swin_tiny_%j.out
# sh sh_ll/train.sh
# python tools/train.py configs/depth/depth_pretrained_h36m.py \
#     --work-dir work_dirs/depth_from_zy_slurm  --gpus 1 --resume-from work_dirs/zy_model/epoch_17.pth


# python tools/train.py configs/depth/depth_pretrained_h36m_SwimBase.py \
#     --work-dir work_dirs/depth_swin2_base  \
#      --resume-from work_dirs /depth_swin2_base/epoch_15.pth \
#     --gpus 1

# python tools/train.py configs/depth/depth_pretrained_h36m.py \
#     --work-dir work_dirs/depth_from_zy_Ba_64  \
#      --resume-from work_dirs/depth_from_zy/epoch_36.pth  \
#     --gpus 1

export CUDA_VISIBLE_DEVICES=0


CONFIG=configs/depth/depth_pretrained_h36m.py
GPUS=1
PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --work-dir work_dirs/test_dist  --launcher pytorch 
    
    #${@:4}

