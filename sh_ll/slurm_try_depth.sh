#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=8 --mem=40GB
#SBATCH --job-name="3D_KP"
# we run on the gpu partition and we allocate 1 titanx gpu
##SBATCH -p gpu --gres=gpu:1 -x hendrixgpu05fl,hendrixgpu06fl,hendrixgpu07fl,hendrixgpu08fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu05fl
##SBATCH -p gpu --gres=gpu:1 -w hendrixgpu01fl,hendrixgpu02fl,hendrixgpu03fl,hendrixgpu04fl,hendrixgpu06fl
##SBATCH -p ml4good --gres=gpu:1 
#SBATCH -p gpu --gres=gpu:1 

#SBATCH --time=7-25:00:00
#SBATCH --mail-type=END,FAIL#SBATCH --mail-user=lilei@di.ku.ku.
#your script, in this case: write the hostname and the ids of the chosen gpus.

#SBATCH --output=./zzslurm_ouput/slurm-depth_base_KP_depth_%j.out


echo $hostname
echo $CUDA_VISIBLE_DEVICES

#SBATCH --output ./slurm_ouput/slurm-depth_swinBase_KP_%j.out
# sh sh_ll/train.sh
export CUDA_VISIBLE_DEVICES = 0

python tools/train.py configs/depth/depth_pretrained_h36m_SwimBase.py \
    --work-dir work_dirs/depth_swin2_base_center96_freezefrom_all\
    --gpus 1
