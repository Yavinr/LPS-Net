import sys
import os
import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from libs.pointops.functions import pointops
import math


class SharedConv_single(nn.Module):

    def __init__(self, CH_in, CH_out, kernal, bn, act=None):

        super(SharedConv_single, self).__init__()

        def calc_conv_para(CH_in, CH_out, kernal):
            stride = max(math.ceil((CH_in - kernal) / (CH_out - 1)), 1)
            if stride * (CH_out - 1) > (CH_in - kernal):
                padding = stride * (CH_out - 1) - (CH_in - kernal)
            else:
                padding = 0
            return stride, padding

        stride, padding = calc_conv_para(CH_in, CH_out, kernal)
        if padding % 2 != 0:
            kernal = kernal + 1
            stride, padding = calc_conv_para(CH_in, CH_out, kernal)
        padding = int(padding / 2)

        self.conv = nn.Conv2d(1, 1, (kernal, 1), stride=(stride, 1),
                              bias=False, padding=(padding, 0))
        if bn == 'single':
            self.bn = nn.BatchNorm2d(1)
        else:
            self.bn = nn.BatchNorm2d(CH_out)

        if act is None:
            self.act = nn.ReLU(inplace=True)
        elif act == 'without_act':
            self.act = None
        else:
            self.act = act

    def forward(self, feature):

        f_shape = feature.shape
        feature = feature.reshape(f_shape[0], 1, f_shape[1], -1)
        feature = self.conv(feature)
        if self.bn.num_features == 1:
            feature = self.bn(feature)
            feature = feature.reshape(f_shape[0], -1, f_shape[2], f_shape[3])
        else:
            feature = feature.reshape(f_shape[0], -1, f_shape[2], f_shape[3])
            feature = self.bn(feature)
        if self.act is not None:
            feature = self.act(feature)
        return feature


class SharedConv(nn.Module):
    def __init__(self, Conv_args, bn, act=None, if_percent=False, args=None):

        super(SharedConv, self).__init__()
        dim_list = Conv_args[0]
        kernel_list = Conv_args[1]
        if len(dim_list) - len(kernel_list) != 1 and args is not None and "k_percent" in args.keys():
            kernel_list = [int(i * args["k_percent"]) for i in dim_list[:-1]]
        assert len(dim_list) - len(kernel_list) == 1, 'SharedConv err'
        self.SharedConv_List = nn.ModuleList()
        for i in range(len(kernel_list)):
            self.SharedConv_List.append(
                SharedConv_single(dim_list[i], dim_list[i + 1], kernel_list[i],
                                  bn, act[i] if act is not None else None))

    def forward(self, feature):
        out = feature
        for i in range(len(self.SharedConv_List)):
            out = self.SharedConv_List[i](out)
        return out


class SharedMLP_single(nn.Module):

    def __init__(self, CH_in, CH_out, bn):

        super(SharedMLP_single, self).__init__()
        self.conv = nn.Conv2d(CH_in, CH_out, (1, 1), stride=(1, 1), bias=False,
                              padding=(0, 0))
        if bn == 'single':
            self.bn = nn.BatchNorm2d(1)
        else:
            self.bn = nn.BatchNorm2d(CH_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, feature):

        feature = self.conv(feature)
        if self.bn.num_features == 1:
            f_shape = feature.shape
            feature = feature.reshape(f_shape[0], 1, f_shape[1], -1)
            feature = self.act(self.bn(feature))
            feature = feature.reshape(f_shape[0], -1, f_shape[2], f_shape[3])
        else:
            feature = self.act(self.bn(feature))
        return feature


class SharedMLP(nn.Module):
    def __init__(self, dim_list, bn):

        super(SharedMLP, self).__init__()
        self.SharedMLP_List = nn.ModuleList()
        for i in range(len(dim_list) - 1):
            self.SharedMLP_List.append(
                SharedMLP_single(dim_list[i], dim_list[i + 1], bn))

    def forward(self, feature):
        out = feature
        for i in range(len(self.SharedMLP_List)):
            out = self.SharedMLP_List[i](out)
        return out


class construct_graph(nn.Module):

    def __init__(self, n_FPS, n_kNN=20):
        super(construct_graph, self).__init__()
        self.n_FPS = n_FPS
        self.n_kNN = n_kNN

    def forward(self, xyz, features):
        # ↓[B, n_FPS],FPS index
        center_idx = pointops.furthestsampling(xyz, self.n_FPS)
        xyz_trans = xyz.transpose(1, 2).contiguous()  # [B, 3, N]
        FPS_xyz = pointops.gathering(  # [B, n_FPS, 3]
            xyz_trans,  # [B, 3, N]
            center_idx  # [B, n_FPS]
        ).transpose(1, 2).contiguous() if self.n_FPS is not None else None

        center_features = pointops.gathering(  # [B, f, n_FPS]
            features,  # [B, f, N]
            center_idx  # [B, n_FPS]
        )

        idx = pointops.knnquery(self.n_kNN, xyz, FPS_xyz)  # [B, n_FPS, n_kNN]
        grouped_xyz = pointops.grouping(xyz_trans, idx)
        # ↓[B, 3, n_FPS, n_kNN]
        grouped_xyz = grouped_xyz - FPS_xyz.transpose(1, 2).unsqueeze(-1)
        # ↓[B, f, n_FPS, n_kNN]
        grouped_features = pointops.grouping(features, idx)
        grouped_features = grouped_features - center_features.unsqueeze(-1)
        # ↓[B, 3+f, n_FPS, n_kNN]
        graph_features = torch.cat([grouped_xyz, grouped_features], dim=1)
        return FPS_xyz, graph_features


class EdgeConv(nn.Module):

    def __init__(self, mlp_specific, bn=True, use_xyz=True):
        super(EdgeConv, self).__init__()
        if use_xyz:
            mlp_specific[0] += 3
        self.SharedMLPs = SharedMLP(mlp_specific, bn=bn)

    def forward(self, graph_features):
        out = self.SharedMLPs(graph_features)
        out = F.max_pool2d(out, kernel_size=[1, out.size(3)])
        return out.squeeze(-1)


class attention(nn.Module):

    def __init__(self, ch_in, ch_hidden, gp):
        super(attention, self).__init__()
        # mid_channels = channels
        assert ch_hidden % gp == 0
        self.gp = gp
        self.q_conv = nn.Conv1d(ch_in, ch_hidden, 1, bias=False, groups=gp)
        self.k_conv = nn.Conv1d(ch_in, ch_hidden, 1, bias=False, groups=gp)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(ch_in, ch_hidden, 1)
        self.trans_conv = nn.Conv1d(ch_hidden, ch_in, 1)
        self.after_norm = nn.BatchNorm1d(ch_in)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature):
        B, ch, nums = feature.size()
        query = self.q_conv(feature)  # [B, ch_hidden, N]
        query = query.reshape(B, self.gp, ch // self.gp, nums)
        query = query.permute(0, 1, 3, 2)  # [B, gp, N, ch_hidden/gp]
        key = self.k_conv(feature)  # [B, ch_hidden, N]
        # ↓[B, gp, ch_hidden/gp, N]
        key = key.reshape(B, self.gp, ch // self.gp, nums)
        value = self.v_conv(feature)  # [B, ch_hidden, N]
        energy = (query @ key).sum(dim=1)  # [B, gp, N, N]→[B, N, N]
        attn = self.softmax(energy)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))
        out = value @ attn
        out = self.act(self.after_norm(self.trans_conv(feature - out)))
        feature = feature + out
        return feature


class upsample(nn.Module):

    def __init__(self, mlp, bn='channel', args=None):
        super(upsample, self).__init__()
        self.mlp = SharedConv(mlp, bn, args=args)

    def forward(self, H_xyz, L_xyz, H_feature, L_feature):
        dist, idx = pointops.nearestneighbor(H_xyz, L_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = pointops.interpolation(L_feature, idx, weight)
        new_features = torch.cat([interpolated_feats, H_feature], dim=1)
        return self.mlp(new_features.unsqueeze(-1)).squeeze(-1)


class NetVLAD(nn.Module):

    def __init__(self, feature_size, cluster_size, VLADNet_ipt_size,
                 args=None, add_batch_norm=True, reshape=True):

        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.reshape = reshape
        self.dropout = args["dropout"]
        self.VLADNet_ipt_size = VLADNet_ipt_size
        self.center = GenerateCenter(VLADNet_ipt_size, cluster_size, feature_size, args)
        self.cluster_a = nn.Parameter(torch.randn(1))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(
                torch.randn(cluster_size) * 1 / torch.sqrt(
                    feature_size))  # attention initialization
            self.bn1 = None

    def forward(self, feature):

        N = feature.shape[-1]
        center = self.center(feature)

        feature = feature.transpose(1, 2).contiguous()  # [B, N, f]
        # [B, N, f] @ [f, cluster_size]
        activation = feature @ center  # [B, N, cluster_size]
        if self.add_batch_norm:
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)  # [B*N, cluster_size]
            activation = activation.view(  # [B, N, cluster_size]
                -1, N, self.cluster_size)
        else:
            activation = activation + self.cluster_biases

        activation = self.softmax(activation)  # [B, N, cluster_size]
        a_sum = activation.sum(-2, keepdim=True)  # [B, 1, cluster_size]
        # [B, 1, cluster_size] * [1, f, cluster_size]→[B, f, cluster_size]
        a = a_sum * center.unsqueeze(0) * self.cluster_a
        activation = activation.transpose(1, 2)  # [B, cluster_size, N]
        # [B, cluster_size, N] @ [B, N, f] = [B, f, cluster_size]
        vlad = (activation @ feature).transpose(1, 2) - a
        vlad = F.normalize(vlad).contiguous()
        if self.reshape:
            return vlad.view((-1, self.cluster_size * self.feature_size)), center
        else:
            return vlad


class GenerateCenter(nn.Module):
    def __init__(self, VLADNet_ipt_size, cluster_size, feature_size,
                 args=None):
        super(GenerateCenter, self).__init__()
        self.VLADNet_ipt_size = VLADNet_ipt_size
        self.cluster_size = cluster_size
        self.feature_size = feature_size
        self.cluster_weights = nn.Parameter(
            torch.randn(feature_size, cluster_size) * 1 / torch.sqrt(torch.tensor(feature_size)))

    def forward(self, feature):
        """
        :param feature: [B, f, N]
        :return: center: [B, f, cluster_size]
        """

        center = self.cluster_weights
        return center


class ContextGating(nn.Module):

    def __init__(self, global_descriptor_dim, feature_size,
                 args, add_batch_norm=True):

        super(ContextGating, self).__init__()
        out_dim = args['opt_descriptor_dim']
        self.BN = nn.BatchNorm1d(out_dim)
        self.add_batch_norm = add_batch_norm
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.BN_gate = nn.BatchNorm1d(out_dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(out_dim) * 1 / torch.sqrt(torch.tensor(out_dim)))
            self.BN_gate = None

        SharedConv_config = [[global_descriptor_dim, out_dim],
                             [args['gate_conv']]]
        act_config = ['without_act']
        self.conv = SharedConv(SharedConv_config, bn='channel', act=act_config, args=args)

    def forward(self, global_descriptor):
        global_descriptor = global_descriptor.unsqueeze(-1).unsqueeze(-1)
        # ↑ global_descriptor[B, 21760, 1, 1]
        global_descriptor = (self.conv(global_descriptor)).squeeze()
        # ↑ global_descriptor[B, out_dim]
        if len(global_descriptor.shape) == 1:
            global_descriptor = global_descriptor.unsqueeze(0)
        return global_descriptor


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        n_FPS = args['n_FPS']
        n_kNN = args['n_kNN']
        gp = args['gp']
        EdgeConv_mlp = args['EdgeConv_mlp']
        upsample_mlp = args['upsample_mlp']
        VLAD_cluster = args['VLAD_cluster']
        VLADNet_ipt_size = args['VLADNet_ipt_size']
        self.construct_graph = nn.ModuleList()
        self.EdgeConv = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.NewVLADNet = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.NetVLAD = nn.ModuleList()
        g_descriptor_dim = 0
        for i in range(len(n_FPS)):
            self.construct_graph.append(construct_graph(n_FPS[i], n_kNN[i]))
            self.EdgeConv.append(EdgeConv(EdgeConv_mlp[i], bn=args['EdgeConv_bn']))
            self.attention.append(attention(EdgeConv_mlp[i][-1], EdgeConv_mlp[i][-1], gp[i]))
            self.upsample.append(upsample(upsample_mlp[i], bn=args['upsample_bn'], args=args))
            self.NetVLAD.append(NetVLAD(upsample_mlp[i][0][-1], VLAD_cluster[i], VLADNet_ipt_size[i], args))
            g_descriptor_dim = g_descriptor_dim + upsample_mlp[i][0][-1] * VLAD_cluster[i]
        self.gate = ContextGating(g_descriptor_dim, upsample_mlp[0][-1], args)

    def forward(self, pc, mode='gate'):
        all_graph_feature = []
        all_FPS_xyz = []
        all_graph_feature.append(pc.transpose(1, 2).contiguous())
        all_FPS_xyz.append(pc)
        for i in range(len(self.construct_graph)):
            FPS_xyz, graph_feature = self.construct_graph[i](all_FPS_xyz[i], all_graph_feature[i])
            all_FPS_xyz.append(FPS_xyz)
            graph_feature = self.EdgeConv[i](graph_feature)
            graph_feature = self.attention[i](graph_feature)
            all_graph_feature.append(graph_feature)
        center_list = []
        for i in range(len(self.construct_graph), 0, -1):
            all_graph_feature[i - 1] = self.upsample[i - 1](
                all_FPS_xyz[i - 1], all_FPS_xyz[i], all_graph_feature[i - 1], all_graph_feature[i])
            all_graph_feature[i], center = self.NetVLAD[i - 1](all_graph_feature[i - 1])
            center_list.append(center)
        global_descriptor = torch.cat(all_graph_feature[-1:0:-1], dim=-1)

        if mode == 'return_feat':
            return all_FPS_xyz, all_graph_feature
        elif mode == 'punish':
            VLAD_punsih = sum([torch.norm(center_, dim=(1, 2)) for center_ in center_list])
            return self.gate(global_descriptor), VLAD_punsih
        elif mode == 'gate':
            return self.gate(global_descriptor)
        else:
            assert False, 'mode err!'


if __name__ == '__main__':
    print('done!')
