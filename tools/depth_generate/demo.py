# from __future__ import print_function
import os
import cPickle as pkl
import numpy as np
import render_model
from smpl.smpl_webuser.serialization import load_model
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def renderImage(model,img_path,camPose,camIntrinsics):

    img = cv2.imread(img_path)
    class cam:
        pass
    cam.rt = cv2.Rodrigues(camPose[0:3,0:3])[0].ravel()
    cam.t = camPose[0:3,3]
    cam.f = np.array([camIntrinsics[0,0],camIntrinsics[1,1]])
    cam.c = camIntrinsics[0:2,2]
    h = int(2*cam.c[1])
    w = int(2*cam.c[0])
    im = (render_model.render_model(model, model.f, w, h, cam, img= None))
    return im

if __name__ == '__main__':
    seq_name = 'courtyard_basketball_00'
    datasetDir = '/mnt/nvme/Dataset/pw3d'
    seqs_name = os.listdir(os.path.join(datasetDir,'sequenceFiles/train'))
    for seq_name in seqs_name:
        seq_name = seq_name[:-4]
        file = os.path.join(datasetDir,'sequenceFiles/train',seq_name+'.pkl')
        seq = pkl.load(open(file,'rb'))
        
        print(seq_name)
        test_img = plt.imread(os.path.join(datasetDir,'imageFiles/', seq_name, 'image_00000.jpg'))
        height, width, _ = test_img.shape
        models = list()
        for iModel in range(0,len(seq['v_template_clothed'])):
            if seq['genders'][iModel] == 'm':
                model = load_model("smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
            else:
                model = load_model("smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl")

            model.betas[:10] = seq['betas'][iModel][:10]
            models.append(model)

        for iFrame in tqdm(range(len(seq['campose_valid'][0]))):
            temp_depth = np.zeros((height, width)) + 100
            for iModel in range(len(seq['campose_valid'])):
                if seq['campose_valid'][iModel][iFrame]:
                    models[iModel].pose[:] = seq['poses'][iModel][iFrame]
                    models[iModel].trans[:] = seq['trans'][iModel][iFrame]
                    img_path = os.path.join(datasetDir,'imageFiles',seq['sequence']+'/image_{:05d}.jpg'.format(iFrame))
                    im = renderImage(models[iModel],img_path,seq['cam_poses'][iFrame],seq['cam_intrinsics'])
                    temp_depth = np.minimum(temp_depth, im)
            temp_depth[temp_depth > 10] = 0
            if np.max(temp_depth) > 0:
                if not os.path.exists(os.path.join(datasetDir,'depthFiles/', seq_name)):
                    os.makedirs(os.path.join(datasetDir,'depthFiles/', seq_name))
                np.save(os.path.join(datasetDir,'depthFiles/', seq_name, 'depth_%05d.npy' % iFrame), temp_depth)
        