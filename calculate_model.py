import argparse
import importlib
import yaml
from thop import profile, clever_format
from utils import *


def get_para():
    parser = argparse.ArgumentParser(
        description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='../NewVLAD/log/LPS_Net_exp_k_1_00_1/code/LPS_Net.yaml', help='config file')
    parser.add_argument('--GPU', type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    cfg["device"] = try_gpu(args.GPU)
    return cfg


def get_model(args):
    Model = importlib.import_module(args["model"])
    net = Model.Network(args)
    net = net.to(args["device"])
    print('model is running on', next(net.parameters()).device)
    return net


def calc_model(net, args):
    tensor = torch.rand(1, 4096, 3)
    tensor = tensor.to(args["device"])
    FLOPs, params = profile(net, inputs=(tensor,))
    FLOPs, params = clever_format([FLOPs, params], "%.3f")
    print('The number of FLOPs is %s' % (FLOPs))
    print('The number of used params is %s' % (params))
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of total parameter: %.2fM" % (total / 1e6))


if __name__ == '__main__':
    args = get_para()
    model = get_model(args)
    calc_model(model, args)
