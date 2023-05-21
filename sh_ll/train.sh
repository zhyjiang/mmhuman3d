


# python tools/train.py configs/depth/depth_pretrained_h36m.py --work-dir work_dirs/depth --gpus 1 --no-validate


# python tools/train.py configs/depth/depth_pretrained_h36m.py --work-dir work_dirs/depth_from_zy  --gpus 1 --resume-from work_dirs/zy_model/epoch_17.pth

# python tools/train.py configs/depth/depth_pretrained_h36m.py \
#     --work-dir work_dirs/depth_swin2_base  \
#     --gpus 1

# python tools/train.py configs/depth/depth_pretrained_h36m.py \
#     --work-dir work_dirs/depth_from_zy_Ba_64  \
#     --resume-from work_dirs/depth_from_zy/epoch_36.pth  \
#     --gpus 1


# python tools/train.py configs/depth/depth_pretrained_h36m_SwimBase_KP.py \
    # --work-dir work_dirs/depth_from_Base_KP  \
    # --gpus 1

# python tools/train.py configs/depth/depth_pretrained_h36m_SwimBase.py \
#     --work-dir work_dirs/depth_swin2_base_center96  \
    
#     --gpus 1


export CUDA_VISIBLE_DEVICES=0
python tools/train.py configs/depth/depth_pretrained_3DPW_SwimTiny.py \
    --work-dir work_dirs/Depth_3DPW  \
    --gpus 1
