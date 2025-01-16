import time, os, sys, subprocess
import numpy as np
import cv2
import torch
from torchvision import transforms
from .nets import S3FDNet
from .box_utils import nms_

PATH_WEIGHT = 'model/faceDetector/s3fd/sfd_face.pth'
if os.path.isfile(PATH_WEIGHT) == False:
    Link = "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt"
    cmd = "gdown --id %s -O %s"%(Link, PATH_WEIGHT)
    subprocess.call(cmd, shell=True, stdout=None)
img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')


class S3FD():

    def __init__(self, device='cpu'):

        tstamp = time.time()
        self.device = device

        # print('[S3FD] loading with', self.device)
        self.net = S3FDNet(device=self.device).to(self.device)
        PATH = os.path.join('/home/daoyuan_mm/data_juicer/data_juicer/my_pretrained_method/Light-ASD', PATH_WEIGHT)
        state_dict = torch.load(PATH, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        # print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))
    
    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype('float32')
                scaled_img -= img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(next(self.net.parameters()).device)
                y = self.net(x)

                detections = y.data
                scale = torch.Tensor([w, h, w, h])

                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] > conf_th:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

            keep = nms_(bboxes, 0.1)
            bboxes = bboxes[keep]

        return bboxes
