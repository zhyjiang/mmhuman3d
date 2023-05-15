_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True

# evaluate
evaluation = dict(interval=4, metric=['pa-mpjpe', 'mpjpe'])

img_res = 384

save_vis_folder = "vis/depth_pretrained_h36m_SwimBase_KP_mpjpeloss"
# optimizer
optimizer = dict(
    backbone=dict(type='Adam', lr=1.0e-4),
        # backbone=dict(type='Adam', lr=2.0e-4),

    head=dict(type='Adam', lr=4.0e-4),
            # head=dict(type='Adam', lr=2.0e-4),

)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Fixed', by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=200)

log_config = dict(
    interval=25, hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)

# model settings
width = 32
downsample = False
use_conv = True

find_unused_parameters = True

model = dict(
    type='ImageBodyKPModelEstimator',
    # type='ImageBodyModelEstimator',

    backbone=dict(
        type='DepthPretrained',
        path='data/checkpoints/dpt_swin2_base_384.pt',
        backbone="swin2b24_384", #swin2b24_384 swin2t16_256
        non_negative=True,
        ),
    head=dict(
        type='SimpleHeadKP',
        num_joints=17,
        num_input_features=256,
    ),
    body_model_train=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        # keypoint_approximate=True,
        # extra_joints_regressor='data/body_models/J_regressor_extra.npy'
        joints_regressor='data/body_models/J_regressor_h36m.npy',
        ),
    body_model_test=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        joints_regressor='data/body_models/J_regressor_h36m.npy'),
    img_res=img_res,
    convention='h36m',
    loss_keypoints3d=dict(type='PoseRegLoss', loss_weight=1),
    ## 
    loss_keypoints2d=dict(type='MSELoss', loss_weight=1),
    loss_centermap=dict(type='MSELoss', loss_weight=1),

    # loss_smpl_pose=dict(type='MSELoss', loss_weight=60),
    # loss_smpl_betas=dict(type='MSELoss', loss_weight=60 * 0.001),

    # loss_segm_mask=dict(type='CrossEntropyLoss', loss_weight=60),
    # loss_camera=dict(type='CameraPriorLoss', loss_weight=1),
    test_vis=True,
)

# dataset settings
dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[255 / 2.0, 255 / 2.0, 255 / 2.0], std=[255 / 2.0, 255 / 2.0, 255 / 2.0], to_rgb=True)
# img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)
data_keys = [
    'has_smpl', 'has_keypoints3d', 'has_keypoints2d', 'smpl_body_pose',
    'smpl_global_orient', 'smpl_betas', 'smpl_transl', 'keypoints2d',
    'keypoints3d', 'sample_idx', 'bbox_xywh', 'centermap'
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.4),
    # dict(
    #     type='SyntheticOcclusion',
    #     occluders_file='data/occluders/pascal_occluders.npy'),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='h36m'),
    dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    # dict(type='GenerateCenterTarget', img_res=img_res, heatmap_size=(64, 64), sigma=3, root_id=0),
    
    dict(type='GenerateCenterTarget', img_res=img_res, heatmap_size=(96, 96), sigma=3, root_id=0),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', 'sample_idx'],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

inference_pipeline = [
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img', 'sample_idx'],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

data = dict(
    samples_per_gpu=12, # 24--> 15000MiB, 32--> 25000MiB, 64--> 39000MiB
    workers_per_gpu=8,
    train=dict(
        # type='MixedDataset',
        # configs=[
        #     dict(
        #         type=dataset_type,
        #         dataset_name='h36m',
        #         data_prefix='data',
        #         pipeline=train_pipeline,
        #         whole_image=True,
        #         convention='smpl_24',
        #         ann_file='sample.npz'),
        # ],
        # partition=[1.0],
        type=dataset_type,
        dataset_name='h36m',
        data_prefix='data',
        pipeline=train_pipeline,
        whole_image=True,
        convention='h36m',
        ann_file='h36m_train.npz'
    ),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        # dataset_name='pw3d',
        # data_prefix='data',
        # pipeline=test_pipeline,
        # ann_file='pw3d_test.npz',
        dataset_name='h36m',
        data_prefix='data',
        pipeline=test_pipeline,
        whole_image=True,
        ann_file='sample.npz'
    ),
    val=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        # dataset_name='pw3d',
        # data_prefix='data',
        # pipeline=test_pipeline,
        # ann_file='pw3d_test.npz',
        dataset_name='h36m',
        data_prefix='data',
        pipeline=test_pipeline,
        whole_image=True,
        ann_file='h36m_valid_protocol1.npz'
        ),
)
