import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .box_utils import Detect, PriorBox


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class S3FDNet(nn.Module):

    def __init__(self, device='cpu'):
        super(S3FDNet, self).__init__()
        self.device = device

        self.vgg = nn.ModuleList([
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, 3, 1, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1, 1),
            nn.ReLU(inplace=True),
        ])

        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)

        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 2, padding=1),
            nn.Conv2d(512, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 2, padding=1),
        ])
        
        self.loc = nn.ModuleList([
            nn.Conv2d(256, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(1024, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(256, 4, 3, 1, padding=1),
        ])

        self.conf = nn.ModuleList([
            nn.Conv2d(256, 4, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(1024, 2, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(256, 2, 3, 1, padding=1),
        ])

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect()

    def forward(self, x):
        size = x.size()[2:]
        sources = list()
        loc = list()
        conf = list()

        for k in range(16):
            x = self.vgg[k](x)
        s = self.L2Norm3_3(x)
        sources.append(s)

        for k in range(16, 23):
            x = self.vgg[k](x)
        s = self.L2Norm4_3(x)
        sources.append(s)

        for k in range(23, 30):
            x = self.vgg[k](x)
        s = self.L2Norm5_3(x)
        sources.append(s)

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])

        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)

        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())

        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())

        features_maps = []
        for i in range(len(loc)):
            feat = []
            feat += [loc[i].size(1), loc[i].size(2)]
            features_maps += [feat]

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        with torch.no_grad():
            self.priorbox = PriorBox(size, features_maps)
            self.priors = self.priorbox.forward()

        output = self.detect.forward(
            loc.view(loc.size(0), -1, 4),
            self.softmax(conf.view(conf.size(0), -1, 2)),
            self.priors.type(type(x.data)).to(next(self.extras.parameters()).device)
        )

        return output
