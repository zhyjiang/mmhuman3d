import numpy as np
import torch
from torch import nn
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.resnet import BasicBlock


class MixedKPHead(BaseModule):

    def __init__(self,
                 num_joints=24,
                 num_input_features=256,
                 num_camera_params=3,
                 has_keypoint2dhead=False,
                 has_bbox3dhead=False,
                 ):
        super(MixedKPHead, self).__init__()
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

        self.keypoint_head = nn.Sequential(
            nn.Conv2d(num_input_features, num_input_features, 3, 2, 1),
            nn.BatchNorm2d(num_input_features, momentum=0.1),
            nn.ReLU(inplace=True),
            BasicBlock(num_input_features, num_input_features),
            nn.Conv2d(num_input_features, num_input_features, kernel_size=3, padding=1)
        )
        self.keypoint_final_layer = nn.Conv2d(num_input_features, num_joints*3, 1)

        self.camera_head = nn.Sequential(
            nn.Conv2d(num_input_features, num_input_features, 3, 2, 1),
            nn.BatchNorm2d(num_input_features, momentum=0.1),
            nn.ReLU(inplace=True),
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
        
        pred_keypoint_3d = self.keypoint_head(x['path_1'])
        pred_cam = self.camera_head(x['path_1'])
        pred_keypoint_3d = self.keypoint_final_layer(pred_keypoint_3d)
        
        pred_keypoint_2d = None
        if self.has_keypoint2dhead:
            pred_keypoint_2d = self.keypoint2d_head(x['path_2'])
        
        pred_bbox3d = None
        if self.has_bbox3dhead:
            pred_bbox3d = self.bbox3d_head(x['path_1'])
     
        output = {
            # 'pred_pose': pred_rotmat,
            # 'pred_shape': pred_shape,
            'pred_KP': pred_keypoint_3d,
            'pred_cam': pred_cam,
            'center_heatmap': center_heatmap.squeeze(),
            'pred_KP2d': pred_keypoint_2d,
            'pred_bbox3d': pred_bbox3d
        }
        return output
