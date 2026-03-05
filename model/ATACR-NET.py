import torch
import torch.nn as nn
import torch.nn.init as init
import math

from torch.autograd import Variable
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from graph.tools import get_groups

from model.lib import ST_RenovateNet
from model.tcn import MultiScale_TemporalModeling, PointWiseTCN, DeTGC, init_param

# TCN相关的导入和常量定义
LEAKY_ALPHA = 0.1


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        # 在空间维度上,将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 包含两层全连接,先降维,后升维。最后接一个sigmoid函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # (B,C,H,W)
        B, C, _, _ = x.size()
        # Squeeze: (B,C,H,W)-->avg_pool-->(B,C,1,1)-->view-->(B,C)
        y = self.avg_pool(x).view(B, C)
        # Excitation: (B,C)-->fc-->(B,C)-->(B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)
        # scale: (B,C,H,W) * (B, C, 1, 1) == (B,C,H,W)
        out = x * y
        return out


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            bias=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1), requires_grad=True)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x) + self.bias
        x = self.bn(x)
        return x





class residual_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(residual_conv, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(EdgeConv, self).__init__()

        self.k = k

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, dim=4):  # N, C, T, V

        if dim == 3:
            N, C, L = x.size()
            pass
        else:
            N, C, T, V = x.size()
            x = x.mean(dim=-2, keepdim=False)  # N, C, V

        x = self.get_graph_feature(x, self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]

        if dim == 3:
            pass
        else:
            x = repeat(x, 'n c v -> n c t v', t=T)

        return x

    def knn(self, x, k):

        inner = -2 * torch.matmul(x.transpose(2, 1), x)  # N, V, V
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = - xx - inner - xx.transpose(2, 1)

        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # N, V, k
        return idx

    def get_graph_feature(self, x, k, idx=None):
        N, C, V = x.size()
        if idx is None:
            idx = self.knn(x, k=k)
        device = x.get_device()

        idx_base = torch.arange(0, N, device=device).view(-1, 1, 1) * V

        idx = idx + idx_base
        idx = idx.view(-1)

        x = rearrange(x, 'n c v -> n v c')
        feature = rearrange(x, 'n v c -> (n v) c')[idx, :]
        feature = feature.view(N, V, k, C)
        x = repeat(x, 'n v c -> n v k c', k=k)

        feature = torch.cat((feature - x, x), dim=3)
        feature = rearrange(feature, 'n v k c -> n c v k')

        return feature


class AHA(nn.Module):
    def __init__(self, in_channels, num_layers, CoM):
        super(AHA, self).__init__()

        self.num_layers = num_layers

        groups = get_groups(dataset='NTU', CoM=CoM)

        for i, group in enumerate(groups):
            group = [i - 1 for i in group]
            groups[i] = group

        inter_channels = in_channels // 4

        self.layers = [groups[i] + groups[i + 1] for i in range(len(groups) - 1)]

        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        self.edge_conv = EdgeConv(inter_channels, inter_channels, k=3)

        self.aggregate = nn.Conv1d(inter_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, L, T, V = x.size()

        x_t = x.max(dim=-2, keepdim=False)[0]
        x_t = self.conv_down(x_t)

        x_sampled = []
        for i in range(self.num_layers):
            s_t = x_t[:, :, i, self.layers[i]]
            s_t = s_t.mean(dim=-1, keepdim=True)
            x_sampled.append(s_t)
        x_sampled = torch.cat(x_sampled, dim=2)

        att = self.edge_conv(x_sampled, dim=3)
        att = self.aggregate(att).view(N, C, L, 1, 1)

        out = (x * self.sigmoid(att)).sum(dim=2, keepdim=False)

        return out


class HD_Gconv(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, residual=True, att=False, CoM=21):
        super(HD_Gconv, self).__init__()
        self.num_layers = A.shape[0]
        self.num_subset = A.shape[1]

        self.att = att

        inter_channels = out_channels // (self.num_subset + 1)
        self.adaptive = adaptive

        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            raise ValueError()

        self.conv_down = nn.ModuleList()
        self.conv = nn.ModuleList()
        for i in range(self.num_layers):
            self.conv_d = nn.ModuleList()
            self.conv_down.append(nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, kernel_size=1),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True)
            ))
            for j in range(self.num_subset):
                self.conv_d.append(nn.Sequential(
                    nn.Conv2d(inter_channels, inter_channels, kernel_size=1),
                    nn.BatchNorm2d(inter_channels)
                ))

            self.conv_d.append(EdgeConv(inter_channels, inter_channels, k=5))
            self.conv.append(self.conv_d)

        if self.att:
            self.aha = AHA(out_channels, num_layers=self.num_layers, CoM=CoM)

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        self.bn = nn.BatchNorm2d(out_channels)

        # 7개 conv layer
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):

        A = self.PA

        out = []
        for i in range(self.num_layers):
            y = []
            x_down = self.conv_down[i](x)
            for j in range(self.num_subset):
                z = torch.einsum('n c t u, v u -> n c t v', x_down, A[i, j])
                z = self.conv[i][j](z)
                y.append(z)
            y_edge = self.conv[i][-1](x_down)
            y.append(y_edge)
            y = torch.cat(y, dim=1)

            out.append(y)

        out = torch.stack(out, dim=2)
        if self.att:
            out = self.aha(out)
        else:
            out = out.sum(dim=2, keepdim=False)

        out = self.bn(out)

        out += self.down(x)
        out = self.relu(out)

        return out


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True,
                 kernel_size=5, att=True, CoM=21, eta=4, num_frame=64):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = HD_Gconv(in_channels, out_channels, A, adaptive=adaptive, att=att, CoM=CoM)

        # 直接使用TCN的多尺度时间建模模块
        # 完全复制tcn中Basic_Block的调用方式
        self.tcn1 = MultiScale_TemporalModeling(
            out_channels,
            out_channels,
            eta,
            stride=stride,
            num_scale=4,  # 固定为4，与tcn中Basic_Block一致
            num_frame=num_frame
        )

        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = residual_conv(in_channels, out_channels, kernel_size=1, stride=stride)

        self.se = SEAttention(channel=out_channels)  # 最后用SE筛选关键特征

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        # 使用 SE 注意力机制融合两个特征
        out = self.se(y)  # 筛选最关键的特征

        return out


class Model(nn.Module):
    def build_basic_blocks(self):
        A, CoM = self.graph.A
        base_channels = 64
        # TCN时间建模参数 (参考degcn设置)
        # 需要为每一层计算正确的num_frame，考虑stride的影响
        base_frame = self.num_frame

        self.l1 = TCN_GCN_unit(3, base_channels, A, residual=False, adaptive=self.adaptive, att=False, CoM=CoM,
                              eta=self.eta, num_frame=base_frame)
        self.l2 = TCN_GCN_unit(base_channels, base_channels, A, adaptive=self.adaptive, CoM=CoM,
                              eta=self.eta, num_frame=base_frame)
        self.l3 = TCN_GCN_unit(base_channels, base_channels, A, adaptive=self.adaptive, CoM=CoM,
                              eta=self.eta, num_frame=base_frame)
        self.l4 = TCN_GCN_unit(base_channels, base_channels, A, adaptive=self.adaptive, CoM=CoM,
                              eta=self.eta, num_frame=base_frame)
        self.l5 = TCN_GCN_unit(base_channels, base_channels * 2, A, stride=2, adaptive=self.adaptive, CoM=CoM,
                              eta=self.eta, num_frame=base_frame)
        self.l6 = TCN_GCN_unit(base_channels * 2, base_channels * 2, A, adaptive=self.adaptive, CoM=CoM,
                              eta=self.eta, num_frame=base_frame//2)
        self.l7 = TCN_GCN_unit(base_channels * 2, base_channels * 2, A, adaptive=self.adaptive, CoM=CoM,
                              eta=self.eta, num_frame=base_frame//2)
        self.l8 = TCN_GCN_unit(base_channels * 2, base_channels * 4, A, stride=2, adaptive=self.adaptive, CoM=CoM,
                              eta=self.eta, num_frame=base_frame//2)
        self.l9 = TCN_GCN_unit(base_channels * 4, base_channels * 4, A, adaptive=self.adaptive, CoM=CoM,
                              eta=self.eta, num_frame=base_frame//4)
        self.l10 = TCN_GCN_unit(base_channels * 4, base_channels * 4, A, adaptive=self.adaptive, CoM=CoM,
                               eta=self.eta, num_frame=base_frame//4)

    def build_cl_blocks(self):
        if self.cl_mode == "ST-Multi-Level":
            self.ren_low = ST_RenovateNet(self.base_channel, self.num_frame, self.num_point, self.num_person,
                                          n_class=self.num_class, version=self.cl_version,
                                          pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_mid = ST_RenovateNet(self.base_channel * 2, self.num_frame // 2, self.num_point, self.num_person,
                                          n_class=self.num_class, version=self.cl_version,
                                          pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_high = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person,
                                           n_class=self.num_class, version=self.cl_version,
                                           pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_fin = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person,
                                          n_class=self.num_class, version=self.cl_version,
                                          pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
        else:
            raise KeyError(f"no such Contrastive Learning Mode {self.cl_mode}")

    def __init__(self,
                 # Base Params
                 num_class=60, num_point=25, num_frame=64, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 base_channel=64, drop_out=0, adaptive=True,
                 # Module Params
                 cl_mode=None, multi_cl_weights=[1, 1, 1, 1], cl_version='V0', pred_threshold=0, use_p_map=True,
                 # TCN Temporal Modeling Params (参考degcn设置)
                 eta=4,
                 ):
        super(Model, self).__init__()

        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame
        self.num_person = num_person
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        self.in_channels = in_channels
        self.base_channel = base_channel
        self.drop_out = nn.Dropout(drop_out) if drop_out else lambda x: x
        self.adaptive = adaptive
        self.cl_mode = cl_mode
        self.multi_cl_weights = multi_cl_weights
        self.cl_version = cl_version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map
        # TCN temporal modeling parameters
        self.eta = eta

        self.dataset = 'NTU' if num_point == 25 else 'UCLA'
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.build_basic_blocks()

        if self.cl_mode is not None:
            self.build_cl_blocks()

        self.fc = nn.Linear(self.base_channel * 4, self.num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def get_hidden_feat(self, x, pooling=True, raw=False):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # First stage
        x = self.l1(x)

        # Second stage
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        # Third stage
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)

        # Forth stage
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)

        if raw:
            return x

        if pooling:
            return x.mean(3).mean(1)
        else:
            return x.mean(1)

    def get_ST_Multi_Level_cl_output(self, x, feat_low, feat_mid, feat_high, feat_fin, label):
        logits = self.fc(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss = cl_low * self.multi_cl_weights[0] + cl_mid * self.multi_cl_weights[1] + \
                  cl_high * self.multi_cl_weights[2] + cl_fin * self.multi_cl_weights[3]
        return logits, cl_loss

    def forward(self, x, label=None, get_cl_loss=False, get_hidden_feat=False, **kwargs):

        if get_hidden_feat:
            return self.get_hidden_feat(x)

        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        # x shape: torch.Size([128, 3, 64, 25])
        x = self.l1(x)
        feat_low = x.clone()

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        feat_mid = x.clone()

        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        feat_high = x.clone()

        x = self.l9(x)
        x = self.l10(x)
        feat_fin = x.clone()

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        if get_cl_loss and self.cl_mode == "ST-Multi-Level":
            return self.get_ST_Multi_Level_cl_output(x, feat_low, feat_mid, feat_high, feat_fin, label)

        return self.fc(x)
