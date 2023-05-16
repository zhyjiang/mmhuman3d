from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2

import torch
import torch.nn.functional as F

import mmhuman3d.core.visualization.visualize_smpl as visualize_smpl
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
from mmhuman3d.core.visualization.visualize_keypoints3d import visualize_kp3d
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idx
from mmhuman3d.models.utils import FitsDict
from mmhuman3d.utils.geometry import (
    batch_rodrigues,
    estimate_translation,
    project_points,
    rotation_matrix_to_angle_axis,
)
from ..backbones.builder import build_backbone
from ..body_models.builder import build_body_model
from ..discriminators.builder import build_discriminator
from ..heads.builder import build_head
from ..losses.builder import build_loss
from ..necks.builder import build_neck
from ..registrants.builder import build_registrant
from .base_architecture import BaseArchitecture
import os


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class BodyModelKPEstimator(BaseArchitecture, metaclass=ABCMeta):
    """BodyModelEstimator Architecture.

    Args:
        backbone (dict | None, optional): Backbone config dict. Default: None.
        neck (dict | None, optional): Neck config dict. Default: None
        head (dict | None, optional): Regressor config dict. Default: None.
        disc (dict | None, optional): Discriminator config dict.
            Default: None.
        registration (dict | None, optional): Registration config dict.
            Default: None.
        body_model_train (dict | None, optional): SMPL config dict during
            training. Default: None.
        body_model_test (dict | None, optional): SMPL config dict during
            test. Default: None.
        convention (str, optional): Keypoints convention. Default: "human_data"
        loss_keypoints2d (dict | None, optional): Losses config dict for
            2D keypoints. Default: None.
        loss_keypoints3d (dict | None, optional): Losses config dict for
            3D keypoints. Default: None.
        loss_vertex (dict | None, optional): Losses config dict for mesh
            vertices. Default: None
        loss_smpl_pose (dict | None, optional): Losses config dict for smpl
            pose. Default: None
        loss_smpl_betas (dict | None, optional): Losses config dict for smpl
            betas. Default: None
        loss_camera (dict | None, optional): Losses config dict for predicted
            camera parameters. Default: None
        loss_adv (dict | None, optional): Losses config for adversial
            training. Default: None.
        loss_segm_mask (dict | None, optional): Losses config for predicted
        part segmentation. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 backbone: Optional[Union[dict, None]] = None,
                 img_res: Optional[int] = 256,
                 test_vis: Optional[bool] = False,
                vis_folder: Optional[str] = None,

                 neck: Optional[Union[dict, None]] = None,
                 head: Optional[Union[dict, None]] = None,
                 disc: Optional[Union[dict, None]] = None,
                 registration: Optional[Union[dict, None]] = None,
                 body_model_train: Optional[Union[dict, None]] = None,
                 body_model_test: Optional[Union[dict, None]] = None,
                 convention: Optional[str] = 'human_data',
                 loss_centermap: Optional[Union[dict, None]] = None,
                 loss_keypoints2d: Optional[Union[dict, None]] = None,
                 loss_keypoints3d: Optional[Union[dict, None]] = None,
                 loss_vertex: Optional[Union[dict, None]] = None,
                #  loss_smpl_pose: Optional[Union[dict, None]] = None,
                #  loss_smpl_betas: Optional[Union[dict, None]] = None,
                 loss_camera: Optional[Union[dict, None]] = None,
                 loss_adv: Optional[Union[dict, None]] = None,
                 loss_segm_mask: Optional[Union[dict, None]] = None,
                 init_cfg: Optional[Union[list, dict, None]] = None):
        super(BodyModelKPEstimator, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)
        self.disc = build_discriminator(disc)

        # import ipdb; ipdb.set_trace()
        self.body_model_train = build_body_model(body_model_train)
        self.body_model_test = build_body_model(body_model_test)
        self.convention = convention
        self.img_res = img_res
        self.test_vis = test_vis
        self.vis_folder = vis_folder
        if os.path.exists(self.vis_folder):
            os.makedirs(self.vis_folder)
            

        if self.test_vis:
            self.vis_train_id =  len(glob.glob("train_*.jpg"))
            self.vis_test_id = len(glob.glob("test_*.jpg"))
            self.vis_gap_train = 0
            self.vis_gap_test = 0

        # import ipdb; ipdb.set_trace()

        # TODO: support HMR+

        self.registration = registration
        if registration is not None:
            self.fits_dict = FitsDict(fits='static')
            self.registration_mode = self.registration['mode']
            self.registrant = build_registrant(registration['registrant'])
        else:
            self.registrant = None

        self.loss_keypoints2d = build_loss(loss_keypoints2d)
        self.loss_keypoints3d = build_loss(loss_keypoints3d)

        self.loss_vertex = build_loss(loss_vertex)
        # self.loss_smpl_pose = build_loss(loss_smpl_pose)
        # self.loss_smpl_betas = build_loss(loss_smpl_betas)

        self.loss_adv = build_loss(loss_adv)
        self.loss_camera = build_loss(loss_camera)
        self.loss_segm_mask = build_loss(loss_segm_mask)
        self.loss_centermap = build_loss(loss_centermap)
        set_requires_grad(self.body_model_train, False)
        set_requires_grad(self.body_model_test, False)
    
    def val_step(self, data_batch):
        if self.backbone is not None:
            img = data_batch['img']
            features = self.backbone(img)
        else:
            features = data_batch['features']

        if self.neck is not None:
            features = self.neck(features)

        predictions = self.head(features)
        return predictions

    def train_step(self, data_batch, optimizer, **kwargs):
        """Train step function.

        In this function, the detector will finish the train step following
        the pipeline:
        1. get fake and real SMPL parameters
        2. optimize discriminator (if have)
        3. optimize generator
        If `self.train_cfg.disc_step > 1`, the train step will contain multiple
        iterations for optimizing discriminator with different input data and
        only one iteration for optimizing generator after `disc_step`
        iterations for discriminator.
        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (dict[torch.optim.Optimizer]): Dict with optimizers for
                generator and discriminator (if have).
        Returns:
            outputs (dict): Dict with loss, information for logger,
            the number of samples.
        """
        if self.backbone is not None:
            img = data_batch['img']
            features = self.backbone(img)
        else:
            features = data_batch['features']

        if self.neck is not None:
            features = self.neck(features)

        predictions = self.head(features)
        targets = self.prepare_targets(data_batch)
        
        # optimize discriminator (if have)
        if self.disc is not None:
            self.optimize_discrinimator(predictions, data_batch, optimizer)

        losses = self.compute_losses(predictions, targets)
        # optimizer generator part
        if self.disc is not None:
            adv_loss = self.optimize_generator(predictions)
            losses.update(adv_loss)

        loss, log_vars = self._parse_losses(losses)
        for key in optimizer.keys():
            optimizer[key].zero_grad()
        loss.backward()
        for key in optimizer.keys():
            optimizer[key].step()

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))
        return outputs


    def optimize_discrinimator(self, predictions: dict, data_batch: dict,
                               optimizer: dict):
        """Optimize discrinimator during adversarial training."""
        set_requires_grad(self.disc, True)
        fake_data = self.make_fake_data(predictions, requires_grad=False)
        real_data = self.make_real_data(data_batch)
        fake_score = self.disc(fake_data)
        real_score = self.disc(real_data)

        disc_losses = {}
        disc_losses['real_loss'] = self.loss_adv(
            real_score, target_is_real=True, is_disc=True)
        disc_losses['fake_loss'] = self.loss_adv(
            fake_score, target_is_real=False, is_disc=True)
        loss_disc, log_vars_d = self._parse_losses(disc_losses)

        optimizer['disc'].zero_grad()
        loss_disc.backward()
        optimizer['disc'].step()

    def optimize_generator(self, predictions: dict):
        """Optimize generator during adversarial training."""
        set_requires_grad(self.disc, False)
        fake_data = self.make_fake_data(predictions, requires_grad=True)
        pred_score = self.disc(fake_data)
        loss_adv = self.loss_adv(
            pred_score, target_is_real=True, is_disc=False)
        loss = dict(adv_loss=loss_adv)
        return loss
    
    def compute_centermap_loss(
        self,
        pred_centermap: torch.Tensor,
        gt_centermap: torch.Tensor):
        loss = self.loss_centermap(pred_centermap, gt_centermap)
        return loss

    def compute_keypoints3d_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            gt_keypoints3d: torch.Tensor,
            has_keypoints3d: Optional[torch.Tensor] = None):
        """Compute loss for 3d keypoints."""
        keypoints3d_conf = gt_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints3d_conf = keypoints3d_conf.repeat(1, 1, 3)
        # pred_keypoints3d = torch.mean(pred_keypoints3d, dim=1)
        pred_keypoints3d = pred_keypoints3d.float()
        gt_keypoints3d = gt_keypoints3d[:, :, :3].float()

        # currently, only mpi_inf_3dhp and h36m have 3d keypoints
        # both datasets have right_hip_extra and left_hip_extra
        right_hip_idx = get_keypoint_idx('right_hip_extra', self.convention)
        left_hip_idx = get_keypoint_idx('left_hip_extra', self.convention)
        gt_pelvis = (gt_keypoints3d[:, right_hip_idx, :] +
                     gt_keypoints3d[:, left_hip_idx, :]) / 2
        pred_pelvis = (pred_keypoints3d[:, :, right_hip_idx, :] +
                       pred_keypoints3d[:, :, left_hip_idx, :]) / 2

        gt_keypoints3d = gt_keypoints3d - gt_pelvis[:, None, :]
        pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, :, None, :]
        gt_keypoints3d = gt_keypoints3d[:, None, :, :].repeat(1,pred_keypoints3d.shape[1],1,1)
        loss = self.loss_keypoints3d(
            pred_keypoints3d, gt_keypoints3d, reduction_override='none')
    
        keypoints3d_conf = keypoints3d_conf[:, None, :, :].repeat(1,pred_keypoints3d.shape[1],1,1)

        # If has_keypoints3d is not None, then computes the losses on the
        # instances that have ground-truth keypoints3d.
        # But the zero confidence keypoints will be included in mean.
        # Otherwise, only compute the keypoints3d
        # which have positive confidence.

        # has_keypoints3d is None when the key has_keypoints3d
        # is not in the datasets

        # import ipdb; ipdb.set_trace()

        if has_keypoints3d is None:

            valid_pos = keypoints3d_conf > 0
            if keypoints3d_conf[valid_pos].numel() == 0:
                return torch.Tensor([0]).type_as(gt_keypoints3d)
            loss = torch.sum(loss * keypoints3d_conf)
            loss /= keypoints3d_conf[valid_pos].numel()
        else:

            keypoints3d_conf = keypoints3d_conf[has_keypoints3d == 1]
            if keypoints3d_conf.shape[0] == 0:
                return torch.Tensor([0]).type_as(gt_keypoints3d)
            loss = loss[has_keypoints3d == 1]
            loss = (loss * keypoints3d_conf).mean()
        return loss

    def compute_keypoints2d_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            pred_cam: torch.Tensor,
            gt_keypoints2d: torch.Tensor,
            img_res: Optional[int] = 224,
            focal_length: Optional[int] = 5000,
            has_keypoints2d: Optional[torch.Tensor] = None):
        """Compute loss for 2d keypoints."""
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
        pred_keypoints3d = pred_keypoints3d.view(-1, 17, 3)
        pred_keypoints2d = project_points(
            pred_keypoints3d,
            pred_cam,
            focal_length=focal_length,
            img_res=self.img_res)
        pred_keypoints2d = pred_keypoints2d.view(gt_keypoints2d.shape[0], -1, 17, 2)
        pred_keypoints2d = torch.mean(pred_keypoints2d, dim=1)
        # Normalize keypoints to [-1,1]
        # The coordinate origin of pred_keypoints_2d is
        # the center of the input image.
        pred_keypoints2d = 2 * pred_keypoints2d / (self.img_res - 1)
        # The coordinate origin of gt_keypoints_2d is
        # the top left corner of the input image.
        gt_keypoints2d = 2 * gt_keypoints2d / (self.img_res - 1) - 1
        loss = self.loss_keypoints2d(
            pred_keypoints2d, gt_keypoints2d, reduction_override='none')

        # If has_keypoints2d is not None, then computes the losses on the
        # instances that have ground-truth keypoints2d.
        # But the zero confidence keypoints will be included in mean.
        # Otherwise, only compute the keypoints2d
        # which have positive confidence.
        # has_keypoints2d is None when the key has_keypoints2d
        # is not in the datasets

        if has_keypoints2d is None:
            valid_pos = keypoints2d_conf > 0
            if keypoints2d_conf[valid_pos].numel() == 0:
                return torch.Tensor([0]).type_as(gt_keypoints2d)
            loss = torch.sum(loss * keypoints2d_conf)
            loss /= keypoints2d_conf[valid_pos].numel()
        else:
            keypoints2d_conf = keypoints2d_conf[has_keypoints2d == 1]
            if keypoints2d_conf.shape[0] == 0:
                return torch.Tensor([0]).type_as(gt_keypoints2d)
            loss = loss[has_keypoints2d == 1]
            loss = (loss * keypoints2d_conf).mean()

        return loss

    def compute_vertex_loss(self, pred_vertices: torch.Tensor,
                            gt_vertices: torch.Tensor, has_smpl: torch.Tensor):
        """Compute loss for vertices."""
        gt_vertices = gt_vertices.float()
        # conf = has_smpl.float().view(-1, 1, 1)
        conf = conf.repeat(1, gt_vertices.shape[1], gt_vertices.shape[2])
        loss = self.loss_vertex(
            pred_vertices, gt_vertices, reduction_override='none')
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_vertices)
        loss = torch.sum(loss * conf) / conf[valid_pos].numel()
        return loss

    def compute_smpl_pose_loss(self, pred_rotmat: torch.Tensor,
                               gt_pose: torch.Tensor, has_smpl: torch.Tensor):
        """Compute loss for smpl pose."""
        conf = has_smpl.float().view(-1)
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_pose)
        pred_rotmat = pred_rotmat[valid_pos]
        gt_pose = gt_pose[valid_pos]
        conf = conf[valid_pos]
        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
        loss = self.loss_smpl_pose(
            pred_rotmat, gt_rotmat, reduction_override='none')
        loss = loss.view(loss.shape[0], -1).mean(-1)
        loss = torch.mean(loss * conf)
        return loss

    def compute_smpl_betas_loss(self, pred_betas: torch.Tensor,
                                gt_betas: torch.Tensor,
                                has_smpl: torch.Tensor):
        """Compute loss for smpl betas."""
        conf = has_smpl.float().view(-1)
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_betas)
        pred_betas = pred_betas[valid_pos]
        gt_betas = gt_betas[valid_pos]
        conf = conf[valid_pos]
        loss = self.loss_smpl_betas(
            pred_betas, gt_betas, reduction_override='none')
        loss = loss.view(loss.shape[0], -1).mean(-1)
        loss = torch.mean(loss * conf)
        return loss

    def compute_camera_loss(self, cameras: torch.Tensor):
        """Compute loss for predicted camera parameters."""
        loss = self.loss_camera(cameras)
        return loss

    def compute_part_segmentation_loss(self,
                                       pred_heatmap: torch.Tensor,
                                       gt_vertices: torch.Tensor,
                                       gt_keypoints2d: torch.Tensor,
                                       gt_model_joints: torch.Tensor,
                                       has_smpl: torch.Tensor,
                                       img_res: Optional[int] = 224,
                                       focal_length: Optional[int] = 500):
        """Compute loss for part segmentations."""
        device = gt_keypoints2d.device
        gt_keypoints2d_valid = gt_keypoints2d[has_smpl == 1]
        batch_size = gt_keypoints2d_valid.shape[0]

        gt_vertices_valid = gt_vertices[has_smpl == 1]
        gt_model_joints_valid = gt_model_joints[has_smpl == 1]

        if batch_size == 0:
            return torch.Tensor([0]).type_as(gt_keypoints2d)
        gt_cam_t = estimate_translation(
            gt_model_joints_valid,
            gt_keypoints2d_valid,
            focal_length=focal_length,
            img_size=self.img_res,
        )

        K = torch.eye(3)
        K[0, 0] = focal_length
        K[1, 1] = focal_length
        K[2, 2] = 1
        K[0, 2] = self.img_res / 2.
        K[1, 2] = self.img_res / 2.
        K = K[None, :, :]

        R = torch.eye(3)[None, :, :]
        device = gt_keypoints2d.device
        gt_sem_mask = visualize_smpl.render_smpl(
            verts=gt_vertices_valid,
            R=R,
            K=K,
            T=gt_cam_t,
            render_choice='part_silhouette',
            resolution=self.img_res,
            return_tensor=True,
            body_model=self.body_model_train,
            device=device,
            in_ndc=False,
            convention='pytorch3d',
            projection='perspective',
            no_grad=True,
            batch_size=batch_size,
            verbose=False,
        )
        gt_sem_mask = torch.flip(gt_sem_mask, [1, 2]).squeeze(-1).detach()
        pred_heatmap_valid = pred_heatmap[has_smpl == 1]
        ph, pw = pred_heatmap_valid.size(2), pred_heatmap_valid.size(3)
        h, w = gt_sem_mask.size(1), gt_sem_mask.size(2)
        if ph != h or pw != w:
            pred_heatmap_valid = F.interpolate(
                input=pred_heatmap_valid, size=(h, w), mode='bilinear')

        loss = self.loss_segm_mask(pred_heatmap_valid, gt_sem_mask)
        return loss

    def compute_losses(self, predictions: dict, targets: dict):
        """Compute losses."""
        batch_size = predictions['pred_cam'].shape[0]
        mask = targets['centermap'] > 0.7
        num_sample = torch.sum(mask.view(mask.shape[0], -1), dim=1)
        if torch.sum(num_sample) / num_sample.shape[0] != num_sample[0]:
            mask = targets['centermap'] >= 1
            num_sample = torch.sum(mask.view(mask.shape[0], -1), dim=1)
        # pred_betas = predictions['pred_shape'].permute(0, 2, 3, 1)[mask, ...].view(-1, 10)
        # pred_pose = predictions['pred_pose'][mask, ...].view(-1, 24, 3, 3)

        pred_cam = predictions['pred_cam'].permute(0, 2, 3, 1)[mask, ...].view(-1, 3)
        pred_centermap = predictions['center_heatmap']
        # import ipdb; ipdb.set_trace()
        pred_keypoints3d = predictions['pred_KP'].permute(0, 2, 3, 1)[mask, ...].view(-1, 3*17)


        gt_keypoints3d = targets['keypoints3d']
        gt_keypoints2d = targets['keypoints2d']
        pred_keypoints3d = pred_keypoints3d.view(batch_size, -1, 17, 3)

        if self.test_vis:
            if self.vis_gap_train % 1000 == 0:
                target_img = (targets['img'][0, :, :, :].permute(1, 2, 0) + 1) / 2.0
                target_img = target_img.cpu().numpy()
                centerpos = int(torch.argmax(targets['centermap'][0]))
                center_x, center_y = (centerpos % 64 * 16, centerpos // 64 * 16)
                target_img = cv2.resize(target_img, (1024, 1024), interpolation = cv2.INTER_AREA)
                target_img = cv2.circle(target_img, (center_x, center_y), 10, (1, 0, 0), -1)
                gt_img = visualize_kp3d(gt_keypoints3d[0:1].cpu().numpy()[:, :, :3], data_source='h36m', return_array=True)[0] / 255.0
                pred_img = visualize_kp3d(pred_keypoints3d[0, 11:12].detach().cpu().numpy(), data_source='h36m', return_array=True)[0] / 255.0
                # plt.imsave('vis/train_%06d.jpg' % self.vis_train_id, 
                        #    np.concatenate([target_img, gt_img, pred_img], axis=1))
                plt.imsave( self.vis_folder +'/train_%06d.jpg' % self.vis_train_id, 
                           np.concatenate([target_img, gt_img, pred_img], axis=1))
                # exit()
                self.vis_train_id += 1
            self.vis_gap_train += 1

        # # TODO: temp. Should we multiply confs here?
        # pred_keypoints3d_mask = pred_output['joint_mask']
        # keypoints3d_mask = keypoints3d_mask * pred_keypoints3d_mask

        # TODO: temp solution
        if 'valid_fit' in targets:
            has_smpl = targets['valid_fit'].view(-1)
            # global_orient = targets['opt_pose'][:, :3].view(-1, 1, 3)
            gt_pose = targets['opt_pose']
            gt_betas = targets['opt_betas']
            gt_vertices = targets['opt_vertices']
        else:
            has_smpl = targets['has_smpl'].view(-1)
            gt_pose = targets['smpl_body_pose']
            global_orient = targets['smpl_global_orient'].view(-1, 1, 3)
            gt_pose = torch.cat((global_orient, gt_pose), dim=1).float()
            gt_betas = targets['smpl_betas'].float()

            # gt_pose N, 72
            if self.body_model_train is not None:
                gt_output = self.body_model_train(
                    betas=gt_betas,
                    body_pose=gt_pose[:, 3:],
                    global_orient=gt_pose[:, :3],
                    num_joints=gt_keypoints2d.shape[1])
                gt_vertices = gt_output['vertices']
                gt_model_joints = gt_output['joints']
        if 'has_keypoints3d' in targets:
            has_keypoints3d = targets['has_keypoints3d'].squeeze(-1)
        else:
            has_keypoints3d = None
        if 'has_keypoints2d' in targets:
            has_keypoints2d = targets['has_keypoints2d'].squeeze(-1)
        else:
            has_keypoints2d = None
        if 'pred_segm_mask' in predictions:
            pred_segm_mask = predictions['pred_segm_mask']
        losses = {}
        if self.loss_centermap is not None:
            losses['centermap_loss'] = self.compute_centermap_loss(pred_centermap,
                                                                   targets['centermap'])
        if self.loss_keypoints3d is not None:
            losses['keypoints3d_loss'] = self.compute_keypoints3d_loss(
                pred_keypoints3d,
                gt_keypoints3d,
                has_keypoints3d=has_keypoints3d)
        if self.loss_keypoints2d is not None:
            losses['keypoints2d_loss'] = self.compute_keypoints2d_loss(
                pred_keypoints3d,
                pred_cam,
                gt_keypoints2d,
                has_keypoints2d=has_keypoints2d)
        if self.loss_camera is not None:
            losses['camera_loss'] = self.compute_camera_loss(pred_cam)
        if self.loss_segm_mask is not None:
            losses['loss_segm_mask'] = self.compute_part_segmentation_loss(
                pred_segm_mask, gt_vertices, gt_keypoints2d, gt_model_joints,
                has_smpl)

        return losses

    @abstractmethod
    def make_fake_data(self, predictions, requires_grad):
        pass

    @abstractmethod
    def make_real_data(self, data_batch):
        pass

    @abstractmethod
    def prepare_targets(self, data_batch):
        pass

    def forward_train(self, **kwargs):
        """Forward function for general training.

        For mesh estimation, we do not use this interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    @abstractmethod
    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        pass


class ImageBodyModelEstimator(BodyModelKPEstimator):

    def make_fake_data(self, predictions: dict, requires_grad: bool):
        pred_cam = predictions['pred_cam']
        pred_pose = predictions['pred_pose']
        pred_betas = predictions['pred_shape']
        if requires_grad:
            fake_data = (pred_cam, pred_pose, pred_betas)
        else:
            fake_data = (pred_cam.detach(), pred_pose.detach(),
                         pred_betas.detach())
        return fake_data

    def make_real_data(self, data_batch: dict):
        transl = data_batch['adv_smpl_transl'].float()
        global_orient = data_batch['adv_smpl_global_orient']
        body_pose = data_batch['adv_smpl_body_pose']
        betas = data_batch['adv_smpl_betas'].float()
        pose = torch.cat((global_orient, body_pose), dim=-1).float()
        real_data = (transl, pose, betas)
        return real_data

    def prepare_targets(self, data_batch: dict):
        # Image Mesh Estimator does not need extra process for ground truth
        return data_batch

    def forward_test(self, img: torch.Tensor, img_metas: dict, **kwargs):
        """Defines the computation performed at every call when testing."""
        if self.backbone is not None:
            features = self.backbone(img)
        else:
            features = kwargs['features']

        if self.neck is not None:
            features = self.neck(features)
        
        predictions = self.head(features)
        pred_centermap = predictions['center_heatmap']
        idx = torch.argmax(pred_centermap.view(pred_centermap.shape[0], -1), dim=1)
        y = idx // pred_centermap.shape[2]
        x = idx % pred_centermap.shape[2]
        f = [i for i in range(pred_centermap.shape[0])]
        pred_pose = predictions['pred_pose'][f, y, x, :, :, :]
        pred_betas = predictions['pred_shape'][f, :, y, x]
        pred_cam = predictions['pred_cam'][f, :, y, x]
        pred_output = self.body_model_test(
            betas=pred_betas,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, 0].unsqueeze(1),
            pose2rot=False)
        

        if self.test_vis:
            if self.vis_gap_test % 1000 == 0:
                target_img = (img[0, :, :, :].permute(1, 2, 0) + 1) / 2.0
                target_img = target_img.cpu().numpy()
                centerpos = int(torch.argmax(pred_centermap[0]))
                center_x, center_y = (centerpos % 64 * 16, centerpos // 64 * 16)
                target_img = cv2.resize(target_img, (1024, 1024), interpolation = cv2.INTER_AREA)
                target_img = cv2.circle(target_img, (center_x, center_y), 10, (1, 0, 0), -1)
                pred_img = visualize_kp3d(torch.mean(pred_output['joints'], dim=0).detach().cpu().numpy()[None, :, :], data_source='h36m', return_array=True)[0] / 255.0
                smpl_img = visualize_smpl_pose(verts=pred_output['vertices'][0:1].cpu(), 
                                            body_model_config=dict(
                                                    type='SMPL',
                                                    keypoint_src='h36m',
                                                    keypoint_dst='h36m',
                                                    model_path='data/body_models',
                                                    joints_regressor='data/body_models/J_regressor_h36m.npy'),
                                            )
                smpl_img = smpl_img.cpu().numpy()[0, :, :, :3]
                plt.imsave(self.vis_folder + '/test_%06d.jpg' % self.vis_test_id, 
                           np.concatenate([target_img, pred_img, smpl_img], axis=1))
                self.vis_test_id += 1
            self.vis_gap_test += 1

        pred_vertices = pred_output['vertices']
        pred_keypoints_3d = pred_output['joints']
        all_preds = {}
        all_preds['keypoints_3d'] = pred_keypoints_3d.detach().cpu().numpy()
        all_preds['smpl_pose'] = pred_pose.detach().cpu().numpy()
        all_preds['smpl_beta'] = pred_betas.detach().cpu().numpy()
        all_preds['camera'] = pred_cam.detach().cpu().numpy()
        all_preds['vertices'] = pred_vertices.detach().cpu().numpy()
        image_path = []
        for img_meta in img_metas:
            image_path.append(img_meta['image_path'])
        all_preds['image_path'] = image_path
        all_preds['image_idx'] = kwargs['sample_idx']
        return all_preds

class ImageBodyKPModelEstimator(BodyModelKPEstimator):

    def make_fake_data(self, predictions: dict, requires_grad: bool):
        pred_cam = predictions['pred_cam']
        # pred_pose = predictions['pred_pose']
        # pred_betas = predictions['pred_shape']
        pre_keypoint_3d = predictions['pred_keypoints_3d']
        if requires_grad:
            fake_data = (pred_cam, pre_keypoint_3d)
        else:
            fake_data = (pred_cam.detach(), pre_keypoint_3d.detach())

        return fake_data

    def make_real_data(self, data_batch: dict):
        transl = data_batch['adv_smpl_transl'].float()
        global_orient = data_batch['adv_smpl_global_orient']
        body_pose = data_batch['adv_smpl_body_pose']
        betas = data_batch['adv_smpl_betas'].float()
        pose = torch.cat((global_orient, body_pose), dim=-1).float()
        real_data = (transl, pose, betas)
        return real_data

    def prepare_targets(self, data_batch: dict):
        # Image Mesh Estimator does not need extra process for ground truth
        return data_batch

    def forward_test(self, img: torch.Tensor, img_metas: dict, **kwargs):
        """Defines the computation performed at every call when testing."""
        if self.backbone is not None:
            features = self.backbone(img)
        else:
            features = kwargs['features']

        if self.neck is not None:
            features = self.neck(features)
        
        predictions = self.head(features)
        pred_centermap = predictions['center_heatmap']
        idx = torch.argmax(pred_centermap.view(pred_centermap.shape[0], -1), dim=1)
        y = idx // pred_centermap.shape[2]
        x = idx % pred_centermap.shape[2]
        f = [i for i in range(pred_centermap.shape[0])]
        # pred_pose = predictions['pred_pose'][f, y, x, :, :, :]
        # pred_betas = predictions['pred_shape'][f, :, y, x]
        pre_keypoint_3d = predictions['pred_KP'][f, :, y, x]
        pred_cam = predictions['pred_cam'][f, :, y, x]
        pred_output = self.body_model_test(
            # betas=pred_betas,
            # body_pose=pred_pose[:, 1:],
            pre_keypoint_3d=pre_keypoint_3d,
            global_orient=pred_cam[:, 0].unsqueeze(1),
            pose2rot=False)
        

        if self.test_vis:
            if self.vis_gap_test % 500 == 0:
                target_img = (img[0, :, :, :].permute(1, 2, 0) + 1) / 2.0
                target_img = target_img.cpu().numpy()
                centerpos = int(torch.argmax(pred_centermap[0]))
                center_x, center_y = (centerpos % 64 * 16, centerpos // 64 * 16)
                target_img = cv2.resize(target_img, (1024, 1024), interpolation = cv2.INTER_AREA)
                target_img = cv2.circle(target_img, (center_x, center_y), 10, (1, 0, 0), -1)
                pred_img = visualize_kp3d(torch.mean(pred_output['joints'], dim=0).detach().cpu().numpy()[None, :, :], data_source='h36m', return_array=True)[0] / 255.0
                # smpl_img = visualize_smpl_pose(verts=pred_output['vertices'][0:1].cpu(), 
                #                             body_model_config=dict(
                #                                     type='SMPL',
                #                                     keypoint_src='h36m',
                #                                     keypoint_dst='h36m',
                #                                     model_path='data/body_models',
                #                                     joints_regressor='data/body_models/J_regressor_h36m.npy'),
                #                             )
                # smpl_img = smpl_img.cpu().numpy()[0, :, :, :3]
                plt.imsave(self.vis_folder + '/test_%06d.jpg' % self.vis_test_id, 
                        #    np.concatenate([target_img, pred_img, smpl_img], axis=1))
                            np.concatenate([target_img, pred_img], axis=1))

                self.vis_test_id += 1
            self.vis_gap_test += 1

        pred_vertices = pred_output['vertices']
        pred_keypoints_3d = pred_output['joints']
        all_preds = {}
        all_preds['keypoints_3d'] = pred_keypoints_3d.detach().cpu().numpy()
        # all_preds['smpl_pose'] = pred_pose.detach().cpu().numpy()
        # all_preds['smpl_beta'] = pred_betas.detach().cpu().numpy()
        all_preds['camera'] = pred_cam.detach().cpu().numpy()
        all_preds['vertices'] = pred_vertices.detach().cpu().numpy()
        image_path = []
        for img_meta in img_metas:
            image_path.append(img_meta['image_path'])
        all_preds['image_path'] = image_path
        all_preds['image_idx'] = kwargs['sample_idx']
        return all_preds


class VideoBodyModelEstimator(BodyModelKPEstimator):

    def make_fake_data(self, predictions: dict, requires_grad: bool):
        B, T = predictions['pred_cam'].shape[:2]
        pred_cam_vec = predictions['pred_cam']
        pred_betas_vec = predictions['pred_shape']
        pred_pose = predictions['pred_pose']
        pred_pose_vec = rotation_matrix_to_angle_axis(pred_pose.view(-1, 3, 3))
        pred_pose_vec = pred_pose_vec.contiguous().view(B, T, -1)
        pred_theta_vec = (pred_cam_vec, pred_pose_vec, pred_betas_vec)
        pred_theta_vec = torch.cat(pred_theta_vec, dim=-1)

        if not requires_grad:
            pred_theta_vec = pred_theta_vec.detach()
        return pred_theta_vec[:, :, 6:75]

    def make_real_data(self, data_batch: dict):
        B, T = data_batch['adv_smpl_transl'].shape[:2]
        transl = data_batch['adv_smpl_transl'].view(B, T, -1)
        global_orient = \
            data_batch['adv_smpl_global_orient'].view(B, T, -1)
        body_pose = data_batch['adv_smpl_body_pose'].view(B, T, -1)
        betas = data_batch['adv_smpl_betas'].view(B, T, -1)
        real_data = (transl, global_orient, body_pose, betas)
        real_data = torch.cat(real_data, dim=-1).float()
        return real_data[:, :, 6:75]

    def prepare_targets(self, data_batch: dict):
        # Video Mesh Estimator needs squeeze first two dimensions
        B, T = data_batch['smpl_body_pose'].shape[:2]

        output = {
            'smpl_body_pose': data_batch['smpl_body_pose'].view(-1, 23, 3),
            'smpl_global_orient': data_batch['smpl_global_orient'].view(-1, 3),
            'smpl_betas': data_batch['smpl_betas'].view(-1, 10),
            'has_smpl': data_batch['has_smpl'].view(-1),
            'keypoints3d': data_batch['keypoints3d'].view(B * T, -1, 4),
            'keypoints2d': data_batch['keypoints2d'].view(B * T, -1, 3)
        }
        return output

    def forward_test(self, img_metas: dict, **kwargs):
        """Defines the computation performed at every call when testing."""
        if self.backbone is not None:
            features = self.backbone(kwargs['img'])
        else:
            features = kwargs['features']

        if self.neck is not None:
            features = self.neck(features)

        B, T = features.shape[:2]
        predictions = self.head(features)
        pred_pose = predictions['pred_pose'].view(-1, 24, 3, 3)
        pred_betas = predictions['pred_shape'].view(-1, 10)
        pred_cam = predictions['pred_cam'].view(-1, 3)

        pred_output = self.body_model_test(
            betas=pred_betas,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, 0].unsqueeze(1),
            pose2rot=False)

        pred_vertices = pred_output['vertices']
        pred_keypoints_3d = pred_output['joints']
        all_preds = {}
        all_preds['keypoints_3d'] = pred_keypoints_3d.detach().cpu().numpy()
        all_preds['smpl_pose'] = pred_pose.detach().cpu().numpy()
        all_preds['smpl_beta'] = pred_betas.detach().cpu().numpy()
        all_preds['camera'] = pred_cam.detach().cpu().numpy()
        all_preds['vertices'] = pred_vertices.detach().cpu().numpy()
        all_preds['image_idx'] = \
            kwargs['sample_idx'].detach().cpu().numpy().reshape((-1))
        return all_preds
