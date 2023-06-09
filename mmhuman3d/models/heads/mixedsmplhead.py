import numpy as np
import torch
from torch import nn
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.resnet import BasicBlock

from mmhuman3d.utils.geometry import rot6dplane_to_rotmat


class MixedSMPLHead(BaseModule):

    def __init__(self,
                 num_joints=24,
                 num_input_features=256,
                 num_camera_params=3,
                 has_keypoint2dhead=False,
                 has_bbox3dhead=False,
                 ):
        super(MixedSMPLHead, self).__init__()
        self.center_head = nn.Sequential(
            nn.Conv2d(num_input_features, num_input_features, 3, 2, 1),
            nn.BatchNorm2d(num_input_features, momentum=0.1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(num_input_features, num_input_features, 3, 2, 1),
            # nn.BatchNorm2d(num_input_features, momentum=0.1),
            # nn.ReLU(inplace=True),
            BasicBlock(num_input_features, num_input_features),
            nn.Conv2d(num_input_features, 1, 1)
        )

        self.smpl_head = nn.Sequential(
            nn.Conv2d(num_input_features, num_input_features, 3, 2, 1),
            nn.BatchNorm2d(num_input_features, momentum=0.1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(num_input_features, num_input_features, 3, 2, 1),
            # nn.BatchNorm2d(num_input_features, momentum=0.1),
            # nn.ReLU(inplace=True),
            BasicBlock(num_input_features, num_input_features),
            BasicBlock(num_input_features, num_input_features)
        )
        self.pose_final_layer = nn.Conv2d(num_input_features, num_joints*6, 1)
        self.shape_final_layer = nn.Conv2d(num_input_features, 10, 1)
        self.camera_head = nn.Sequential(
            nn.Conv2d(num_input_features, num_input_features, 3, 2, 1),
            nn.BatchNorm2d(num_input_features, momentum=0.1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(num_input_features, num_input_features, 3, 2, 1),
            # nn.BatchNorm2d(num_input_features, momentum=0.1),
            # nn.ReLU(inplace=True),
            BasicBlock(num_input_features, num_input_features),
            BasicBlock(num_input_features, num_input_features),
            nn.Conv2d(num_input_features, num_camera_params, 1)
        )
        self.num_joints = num_joints

        self.has_keypoint2dhead = has_keypoint2dhead
        self.has_bbox3dhead = has_bbox3dhead

        if self.has_bbox3dhead:
            self.bbox3d_head = nn.Sequential(
                nn.Conv2d(num_input_features, num_input_features, 3, 1, 1),
                nn.BatchNorm2d(num_input_features, momentum=0.1),
                nn.ReLU(inplace=True),
                BasicBlock(num_input_features, num_input_features),
                nn.Conv2d(num_input_features, 6, 1)
            )
        
        if self.has_keypoint2dhead:
            self.keypoint2d_head = nn.Sequential(
                nn.Conv2d(num_input_features, num_input_features, 3, 1, 1),
                nn.BatchNorm2d(num_input_features, momentum=0.1),
                nn.ReLU(inplace=True),
                BasicBlock(num_input_features, num_input_features),
                nn.Conv2d(num_input_features, num_joints, kernel_size=1)
            )

    def forward(self, x):
        center_heatmap = self.center_head(x['path_1'])
        
        pose_feat = self.smpl_head(x['path_1'])
        pred_pose = self.pose_final_layer(pose_feat)
        pred_shape = self.shape_final_layer(pose_feat)
        pred_cam = self.camera_head(x['path_1'])

        pred_pose = pred_pose.permute(0, 2, 3, 1)
        b, h, w = (pred_pose.shape[0], pred_pose.shape[1], pred_pose.shape[2])
        pred_pose = pred_pose.reshape(b * h * w * self.num_joints, 3, 2)
        pred_rotmat = rot6dplane_to_rotmat(pred_pose).reshape(b, h, w, self.num_joints, 3, 3)

        
        pred_keypoint_2d = None
        if self.has_keypoint2dhead:
            pred_keypoint_2d = self.keypoint2d_head(x['path_2'])
        
        pred_bbox3d = None
        if self.has_bbox3dhead:
            pred_bbox3d = self.bbox3d_head(x['path_1'])
     
        output = {
            # 'pred_pose': pred_rotmat,
            # 'pred_shape': pred_shape,
            'pred_depth': x['depth'],
            'pred_pose': pred_rotmat,
            'pred_shape': pred_shape,
            'pred_cam': pred_cam,
            'center_heatmap': center_heatmap.squeeze(),
            'pred_KP2d': pred_keypoint_2d,
            'pred_bbox3d': pred_bbox3d
        }
        return output
