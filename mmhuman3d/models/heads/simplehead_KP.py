import numpy as np
import torch
from torch import nn
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.resnet import BasicBlock

from mmhuman3d.utils.geometry import rot6dplane_to_rotmat


class SimpleHeadKP(BaseModule):

    def __init__(self,
                 num_joints=24,
                 num_input_features=256,
                 num_camera_params=3,
                 ):
        super(SimpleHeadKP, self).__init__()
        self.center_head = nn.Sequential(
            nn.Conv2d(num_input_features, num_input_features, 3, 2, 1),
            nn.BatchNorm2d(num_input_features, momentum=0.1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(num_input_features, num_input_features, 3, 2, 1),
            # nn.BatchNorm2d(num_input_features, momentum=0.1),
            # nn.ReLU(inplace=True),
            BasicBlock(num_input_features, num_input_features),
            BasicBlock(num_input_features, num_input_features),
            nn.Conv2d(num_input_features, 1, 1)
        )

        # self.smpl_head = nn.Sequential(
        #     nn.Conv2d(num_input_features, num_input_features, 3, 2, 1),
        #     nn.BatchNorm2d(num_input_features, momentum=0.1),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(num_input_features, num_input_features, 3, 2, 1),
        #     # nn.BatchNorm2d(num_input_features, momentum=0.1),
        #     # nn.ReLU(inplace=True),
        #     BasicBlock(num_input_features, num_input_features),
        #     BasicBlock(num_input_features, num_input_features)
        # )
        # self.pose_final_layer = nn.Conv2d(num_input_features, num_joints*6, 1)
        # self.shape_final_layer = nn.Conv2d(num_input_features, 10, 1)

        self.keypoint_head = nn.Sequential(
            nn.Conv2d(num_input_features, num_input_features, 3, 2, 1),
            nn.BatchNorm2d(num_input_features, momentum=0.1),
            nn.ReLU(inplace=True),
            BasicBlock(num_input_features, num_input_features),
            BasicBlock(num_input_features, num_input_features),
            nn.Conv2d(num_input_features, num_input_features, kernel_size=3, padding=1)
        )
        self.keypoint_final_layer = nn.Conv2d(num_input_features, num_joints*3, 1)


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

    def forward(self, x):
        batch_size = x.shape[0]
        center_heatmap = self.center_head(x)
        # pose_feat = self.smpl_head(x)
        # pred_pose = self.pose_final_layer(pose_feat)
        # pred_shape = self.shape_final_layer(pose_feat)
        pre_keypoint_3d = self.keypoint_head(x)
        pred_cam = self.camera_head(x)
        pre_keypoint_3d = self.keypoint_final_layer(pre_keypoint_3d)

        # pred_pose = pred_pose.permute(0, 2, 3, 1)
        # b, h, w = (pred_pose.shape[0], pred_pose.shape[1], pred_pose.shape[2])
        # pred_pose = pred_pose.reshape(b * h * w * self.num_joints, 3, 2)
        # pred_rotmat = rot6dplane_to_rotmat(pred_pose).reshape(b, h, w, self.num_joints, 3, 3)
     
        output = {
            # 'pred_pose': pred_rotmat,
            # 'pred_shape': pred_shape,
            'pred_KP': pre_keypoint_3d,
            'pred_cam': pred_cam,
            'center_heatmap': center_heatmap.squeeze()
        }
        return output
