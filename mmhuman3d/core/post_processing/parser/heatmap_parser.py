import torch


class HeatmapParser(object):
    def __init__(self, nms_kernel):
        assert nms_kernel % 2 == 1
        self.pool = torch.nn.MaxPool2d(
            nms_kernel, 1, nms_kernel // 2
        )

    def nms(self, det):
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det