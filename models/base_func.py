import torch
from torch import nn
import torch.nn.functional as F

def weights_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class EnLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x


class LatLayer(nn.Module):
    def __init__(self, in_channel):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class DSLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0))#, nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class half_DSLayer(nn.Module):
    def __init__(self, in_channel=512):
        super(half_DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel/4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(int(in_channel/4), 1, kernel_size=1, stride=1, padding=0)) #, nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class AugAttentionModule(nn.Module):
    def __init__(self, input_channels=512):
        super(AugAttentionModule, self).__init__()
        self.query_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.key_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.value_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.conv(x)
        x_query = self.query_transform(x).view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        # x_key: C,BHW
        x_key = self.key_transform(x).view(B, C, -1)  # B, C,HW
        # x_value: BHW, C
        x_value = self.value_transform(x).view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        # Aqk为attention_bmm
        attention_bmm = torch.bmm(x_query, x_key)*self.scale # B, HW, HW
        print("-----------------attention_bmm-------------------")
        print(attention_bmm.size())

        # Aatt为attention
        attention = F.softmax(attention_bmm, dim=-1)
        print("-----------------attention-------------------")
        print(attention.size())

        attention_sort = torch.sort(attention_bmm, dim=-1, descending=True)[1]
        print("-----------------attention_sort-------------------")
        print(attention_sort.size())
        # Asort为attention_sort
        attention_sort = torch.sort(attention_sort, dim=-1)[1]
        print("-----------------attention_sort-------------------")
        print(attention_sort.size())

        #####
        attention_positive_num = torch.ones_like(attention).cuda()
        # 去负操作
        attention_positive_num[attention_bmm < 0] = 0
        print("-----------------attention_positive_num-------------------")
        print(attention_positive_num.size())
        # Apm为att_pos_mask
        att_pos_mask = attention_positive_num.clone()

        # 统计正值注意力的数量
        attention_positive_num = torch.sum(attention_positive_num, dim=-1, keepdim=True).expand_as(attention_sort)
        # 创建排序掩码attention_sort的副本
        attention_sort_pos = attention_sort.float().clone()

        # 获取每一行的正索引最大值
        apn = attention_positive_num-1
        # 通过上面的最大值，进行筛选，将负注意力筛除
        attention_sort_pos[attention_sort > apn] = 0
        print("-----------------attention_sort_pos-------------------")
        print(attention_sort_pos.size())
        # Aam = ((+1)^3)*Apm + (1-Apm)
        attention_mask = ((attention_sort_pos+1)**3)*att_pos_mask + (1-att_pos_mask)
        out = torch.bmm(attention*attention_mask, x_value)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        return out+x


class AttLayer(nn.Module):
    def __init__(self, input_channels=512):
        super(AttLayer, self).__init__()
        self.query_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.key_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)

    def correlation(self, x5, seeds):
        # 这个函数的主要功能是计算输入特征 x5 和种子特征 seeds 的归一化相关性映射：
        # 使用种子特征作为卷积核，对输入特征进行逐像素相关性计算。
        # 对相关性结果按通道取均值，并归一化到 [0, 1]。
        # 恢复为二维空间形状，作为输出相关性映射。
        B, C, H5, W5 = x5.size()
        if self.training:
            # 将seeds作为卷积权重，进行卷积
            # B,B,H,W
            correlation_maps = F.conv2d(x5, weight=seeds) 
        else:
            correlation_maps = torch.relu(F.conv2d(x5, weight=seeds))  # B,B,H,W
        
        # B, HW
        correlation_maps = correlation_maps.mean(1).view(B, -1)
        # 对每个样本沿着空间维度计算最小值和最大值
        min_value = torch.min(correlation_maps, dim=1, keepdim=True)[0]
        max_value = torch.max(correlation_maps, dim=1, keepdim=True)[0]
        # 通过最小值和最大值，将每个像素点的相关性值归一化到[0, 1]范围，形状仍为B,HW
        correlation_maps = (correlation_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[B, HW]
        # 形状重塑为B, 1, H, W
        correlation_maps = correlation_maps.view(B, 1, H5, W5)  # shape=[B, 1, H, W]
        return correlation_maps

    def forward(self, x5):
        # x: B,C,H,W
        x5 = self.conv(x5)+x5
        B, C, H5, W5 = x5.size()
        x_query = self.query_transform(x5).view(B, C, -1)
        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C)  # BHW, C
        # x_key: B,C,HW
        x_key = self.key_transform(x5).view(B, C, -1)
        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1)  # C, BHW
        # W = Q^T K: B,HW,HW
        x_w1 = torch.matmul(x_query, x_key) * self.scale # BHW, BHW
        x_w = x_w1.view(B * H5 * W5, B, H5 * W5)
        x_w = torch.max(x_w, -1).values  # BHW, B
        x_w = x_w.mean(-1)
        x_w = x_w.view(B, -1)   # B, HW
        x_w = F.softmax(x_w, dim=-1)  # B, HW

        #####  mine ######
        # 沿x5的1维度（通道维度）进行L2归一化操作，使其每个通道的特征向量具有单位长度
        norm0 = F.normalize(x5, dim=1)
        # x_w维度变为：B，1，HW
        x_w = x_w.unsqueeze(1)
        # 在最后一个维度取最大值，并重塑为 B，1，HW
        x_w_max = torch.max(x_w, -1).values.unsqueeze(2).expand_as(x_w)
        # 创建一个和x_w形状相同的全零张量
        mask = torch.zeros_like(x_w).cuda()
        # 将x_w中等于x_w_max的位置的值设为1，其它位置保持为0
        mask[x_w == x_w_max] = 1
        # 将mask的形状重塑为 B, 1, H, W
        mask = mask.view(B, 1, H5, W5)
        # seeds 形状：B, C, H, W
        seeds = norm0 * mask
        # 在H和W维度上进行求和，并重塑为B,C,1,1
        seeds = seeds.sum(3).sum(2).unsqueeze(2).unsqueeze(3)
        # 计算norm0和seeds的相关性
        # B, 1, H, W
        cormap = self.correlation(norm0, seeds)
        x51 = x5 * cormap
        proto1 = torch.mean(x51, (0, 2, 3), True)
        # x5*proto1+x51即x5*proto1 + x5*cormap
        return x5, proto1, x5*proto1+x51, mask

# CALayer：先全局平均池化，再1*1卷积，最后sigmoid
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
# RCAB：先卷积+ReLU+卷积，然后全局平均池化，得到的结果和原特征相加
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        # 这一段: modules_body = 卷积+ReLU+卷积
        for i in range(2): 
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        # CALayer：先全局平均池化，再1*1卷积，最后sigmoid
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    # RCAB：先卷积+ReLU+卷积，然后全局平均池化，得到的结果和原特征相加
    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale) 
        res += x
        return res

class Edge_Module2_5(nn.Module):
    def __init__(self, in_fea=[128, 512], mid_fea=64):
        super(Edge_Module2_5, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv5 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.rcab = RCAB(mid_fea * 2)

    def forward(self, x2, x5):
        _, _, h, w = x2.size()

        edge2_fea = self.relu(self.conv2(x2))
        edge2 = self.relu(self.conv5_2(edge2_fea))
        edge5_fea = self.relu(self.conv5(x5))
        edge5 = self.relu(self.conv5_5(edge5_fea))

        edge5 = F.interpolate(edge5, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge2, edge5], dim=1)
        # edge先卷积+ReLU+卷积，然后全局平均池化，得到的结果和原edge相加
        # edge通道数为128
        edge = self.rcab(edge)
        return edge

class BasicConv2d(nn.Module):    #很多模块的使用卷积层都是以其为基础，论文中的BConvN
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

#Global Contextual module
class GCM(nn.Module):  # 输入通道首先经过四个卷积层的特征提取，并采用torch.cat()进行连接，最后和输入通道的残差进行相加
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(out_channel, out_channel, kernel_size=(9, 1), padding=(4, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=9, dilation=9)
        )
        self.conv_cat = BasicConv2d(6*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3, x4, x5), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
    
class SpatialAtt(nn.Module):
    # 输入特征，返回经过处理后的该层特征
    def __init__(self, in_channels):
        super(SpatialAtt, self).__init__()
        self.enc_fea_proc = nn.Sequential(
            nn.BatchNorm2d(64, momentum=0.001),
            nn.ReLU(inplace=True),
        )
        self.SpatialAtt = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, the_enc, last_dec):
        [_, _, H, W] = the_enc.size()
        last_dec_1 = F.interpolate(last_dec, size=(H, W), mode='bilinear', align_corners=False)
        spatial_att = self.SpatialAtt(last_dec_1)
        the_enc = self.enc_fea_proc(the_enc)
        the_enc1 = the_enc*spatial_att
        return the_enc1