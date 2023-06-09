import os
import pickle
from typing import List
from functools import reduce

import cv2
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.cameras.camera_parameters import CameraParameter
from mmhuman3d.data.data_converters.builder import DATA_CONVERTERS
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.data_structures.multi_human_data import MultiHumanData
from .base_converter import BaseModeConverter


@DATA_CONVERTERS.register_module()
class Pw3dConverter(BaseModeConverter):
    """3D Poses in the Wild dataset `Recovering Accurate 3D Human Pose in The
    Wild Using IMUs and a Moving Camera' ECCV'2018 More details can be found in
    the `paper.

    <https://virtualhumans.mpi-inf.mpg.de/papers/vonmarcardECCV18/
    vonmarcardECCV18.pdf>`__ .

    Args:
        modes (list): 'test' and/or 'train' for accepted modes
    """

    ACCEPTED_MODES = ['train', 'test','validation']

    def __init__(self, modes: List = []):
        super(Pw3dConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str, enable_multi_human_data=False):
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, smpl, meta
                stored in HumanData() format
        """
        # use HumanData to store all data
        if enable_multi_human_data:
            human_data = MultiHumanData()
        else:
            human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, cam_param_ = [], [], []
        depth_path_ = []
        translation = []
        smpl = {}
        smpl['body_pose'] = []
        smpl['global_orient'] = []
        smpl['betas'] = []
        meta = {}
        meta['gender'] = []
        frame_range = []

        root_path = dataset_path
        # get a list of .pkl files in the directory
        dataset_path = os.path.join(dataset_path, 'sequenceFiles', mode)
        files = [
            os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
            if f.endswith('.pkl')
        ]
        # files = ['data/datasets/pw3d/sequenceFiles/train/courtyard_goodNews_00.pkl']

        # go through all the .pkl files
        for filename in tqdm(files):
            with open(filename, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                smpl_pose = data['poses']
                smpl_betas = data['betas']
                poses2d = data['poses2d']
                global_poses = data['cam_poses']
                genders = data['genders']
                trans = data['trans']
                valid = np.array(data['campose_valid']).astype(np.bool)
                frame_valid = np.bitwise_or.reduce(valid, axis=0)
                temp_body_pose = [[] for _ in range(np.sum(frame_valid))]
                temp_trans = [[] for _ in range(np.sum(frame_valid))]
                temp_global_orient = [[] for _ in range(np.sum(frame_valid))]
                temp_betas = [[] for _ in range(np.sum(frame_valid))]
                temp_meta_gender = [[] for _ in range(np.sum(frame_valid))]
                temp_bbox_xywh_ = [[] for _ in range(np.sum(frame_valid))]

                K = np.array(data['cam_intrinsics'])
                num_people = len(smpl_pose)
                num_frames = len(smpl_pose[0])
                seq_name = str(data['sequence'])
                img_names = np.array([
                    'imageFiles/' + seq_name + f'/image_{str(i).zfill(5)}.jpg'
                    for i in range(num_frames)
                ])

                temp_image_path_ = [[] for _ in range(np.sum(frame_valid))]
                temp_depth_path_ = [[] for _ in range(np.sum(frame_valid))]
                temp_cam_param_ = [[] for _ in range(np.sum(frame_valid))]
                
                # get through all the people in the sequence
                for i in range(num_people):
                    valid_pose = smpl_pose[i][valid[i]]
                    valid_betas = np.tile(smpl_betas[i][:10].reshape(1, -1),
                                          (num_frames, 1))
                    valid_betas = valid_betas[valid[i]]
                    valid_keypoints_2d = poses2d[i][valid[i]]
                    valid_img_names = img_names[valid[i]]
                    valid_global_poses = global_poses[valid[i]]
                    valid_trans = trans[i][valid[i]]
                    valid_idx = np.where(valid[i][frame_valid])[0]
                    gender = genders[i]

                    image_path = valid_img_names[0]
                    image_abs_path = os.path.join(root_path, image_path)
                    h, w, _ = cv2.imread(image_abs_path).shape

                    # consider only valid frames
                    for valid_i in range(valid_pose.shape[0]):
                        keypoints2d = valid_keypoints_2d[valid_i, :, :].T
                        keypoints2d = keypoints2d[keypoints2d[:, 2] > 0, :]
                        bbox_xyxy = [
                            min(keypoints2d[:, 0]),
                            min(keypoints2d[:, 1]),
                            max(keypoints2d[:, 0]),
                            max(keypoints2d[:, 1])
                        ]
                        tran = valid_trans[valid_i]

                        bbox_xyxy = self._bbox_expand(
                            bbox_xyxy, scale_factor=1.2)
                        bbox_xywh = self._xyxy2xywh(bbox_xyxy)

                        image_path = valid_img_names[valid_i]

                        # transform global pose
                        pose = valid_pose[valid_i]
                        extrinsic_param = valid_global_poses[valid_i]
                        R = extrinsic_param[:3, :3]
                        T = extrinsic_param[:3, 3]

                        camera = CameraParameter(H=h, W=w)
                        camera.set_KRT(K, R, T)
                        parameter_dict = camera.to_dict()

                        pose[:3] = cv2.Rodrigues(
                            np.dot(R,
                                   cv2.Rodrigues(pose[:3])[0]))[0].T[0]

                        temp_image_path_[valid_idx[valid_i]].append(image_path)
                        temp_depth_path_[valid_idx[valid_i]].append(image_path.replace('image', 'depth').replace('jpg', 'npy'))
                        temp_bbox_xywh_[valid_idx[valid_i]].append(bbox_xywh)
                        temp_body_pose[valid_idx[valid_i]].append(pose[3:].reshape((23, 3)))
                        temp_trans[valid_idx[valid_i]].append(tran)
                        temp_global_orient[valid_idx[valid_i]].append(pose[:3])
                        temp_betas[valid_idx[valid_i]].append(valid_betas[valid_i])
                        temp_meta_gender[valid_idx[valid_i]].append(gender)
                        temp_cam_param_[valid_idx[valid_i]].append(parameter_dict)
                
                frame_start = len(smpl['body_pose'])
                item_count = 0
                for item in temp_body_pose:
                    frame_range.append([frame_start + item_count, frame_start + item_count + len(item)])
                    item_count += len(item)
                smpl['body_pose'].extend(reduce(lambda a,b:a+b, temp_body_pose))
                smpl['global_orient'].extend(reduce(lambda a,b:a+b, temp_global_orient))
                smpl['betas'].extend(reduce(lambda a,b:a+b, temp_betas))
                meta['gender'].extend(reduce(lambda a,b:a+b, temp_meta_gender))
                bbox_xywh_.extend(reduce(lambda a,b:a+b, temp_bbox_xywh_))
                image_path_.extend(reduce(lambda a,b:a+b, temp_image_path_))
                depth_path_.extend(reduce(lambda a,b:a+b, temp_depth_path_))
                cam_param_.extend(reduce(lambda a,b:a+b, temp_cam_param_))
                translation.extend(reduce(lambda a,b:a+b, temp_trans))

        # change list to np array
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        smpl['body_pose'] = np.array(smpl['body_pose']).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(smpl['global_orient']).reshape(
            (-1, 3))
        smpl['betas'] = np.array(smpl['betas']).reshape((-1, 10))
        meta['gender'] = np.array(meta['gender'])
        frame_range = np.array(frame_range).reshape((-1, 2))
        smpl['transl'] = np.array(translation)

        human_data['image_path'] = image_path_
        human_data['depth_path'] = depth_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['smpl'] = smpl
        human_data['meta'] = meta
        human_data['cam_param'] = cam_param_
        human_data['config'] = 'pw3d'
        human_data['frame_range'] = frame_range

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'pw3d_{}.npz'.format(mode)
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
