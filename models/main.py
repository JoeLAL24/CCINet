import torch
from torch import nn
import torch.nn.functional as F
from models.vgg import VGG_Backbone
from util import *
from models.base_func import *

class Decoder_edge_GCM_SA(nn.Module):
    def __init__(self):
        super(Decoder_edge_GCM_SA, self).__init__()
        self.toplayer = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.latlayer4 = LatLayer(in_channel=512)
        self.latlayer3 = LatLayer(in_channel=256)
        self.latlayer2 = LatLayer(in_channel=128)
        self.latlayer1 = LatLayer(in_channel=64)

        self.enlayer4 = EnLayer()
        self.enlayer3 = EnLayer()
        self.enlayer2 = EnLayer()
        self.enlayer1 = EnLayer()

        self.dslayer4 = DSLayer()
        self.dslayer3 = DSLayer()
        self.dslayer2 = DSLayer()
        self.dslayer1 = DSLayer()

        self.gcm1 = GCM(64, 64)

        self.spatialatt1 = SpatialAtt(64)

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x + y

    def forward(self, weighted_x5, x4, x3, x2, x1, H, W, edge):
        preds = []
        p5 = self.toplayer(weighted_x5)
        p4 = self._upsample_add(p5, self.latlayer4(x4))
        p4 = self.enlayer4(p4)
        _pred = self.dslayer4(p4)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        p3 = self._upsample_add(p4, self.latlayer3(x3))
        p3 = self.enlayer3(p3)
        _pred = self.dslayer3(p3)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        x2 = x2 + edge
        p2 = self._upsample_add(p3, self.latlayer2(x2))
        p2 = self.enlayer2(p2)
        _pred = self.dslayer2(p2)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        # x1(N, 64, 224, 224)   p2(N, 64, 112, 112)
        x1_1 = self.spatialatt1(x1, p2)
        x1_2 = self.gcm1(x1_1)
        p1 = self._upsample_add(p2, self.latlayer1(x1_2))
        _pred = self.dslayer1(p1)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))
        return preds


class DCFMNet(nn.Module):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, mode='train'):
        super(DCFMNet, self).__init__()
        self.gradients = None
        self.backbone = VGG_Backbone()
        self.mode = mode
        self.aug = AugAttentionModule()
        self.fusion = AttLayer(512)
        self.decoder_edge_gcm_sa = Decoder_edge_GCM_SA()
        self.edge_module2_5 = Edge_Module2_5()

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x, gt):
        if self.mode == 'train':
            preds = self._forward(x, gt)
        else:
            with torch.no_grad():
                preds = self._forward(x, gt)

        return preds

    def featextract(self, x):
        x1 = self.backbone.conv1(x)
        x2 = self.backbone.conv2(x1)
        x3 = self.backbone.conv3(x2)
        x4 = self.backbone.conv4(x3)
        x5 = self.backbone.conv5(x4)
        return x5, x4, x3, x2, x1

    def _forward(self, x, gt):
        [B, _, H, W] = x.size()
        x5, x4, x3, x2, x1 = self.featextract(x)
        feat, proto, weighted_x5, cormap = self.fusion(x5)
        feataug = self.aug(weighted_x5)
        edge = self.edge_module2_5(x2, feataug)
        preds = self.decoder_edge_gcm_sa(feataug, x4, x3, x2, x1, H, W, edge)
        if self.training:
            gt = F.interpolate(gt, size=weighted_x5.size()[2:], mode='bilinear', align_corners=False)
            feat_pos, proto_pos, weighted_x5_pos, cormap_pos = self.fusion(x5 * gt)
            feat_neg, proto_neg, weighted_x5_neg, cormap_neg = self.fusion(x5*(1-gt))
            return preds, proto, proto_pos, proto_neg
        return preds


class DCFM(nn.Module):
    def __init__(self, mode='train'):
        super(DCFM, self).__init__()
        set_seed(123)
        self.dcfmnet = DCFMNet()
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode
        self.dcfmnet.set_mode(self.mode)

    def forward(self, x, gt):
        ########## Co-SOD ############
        preds = self.dcfmnet(x, gt)
        return preds

