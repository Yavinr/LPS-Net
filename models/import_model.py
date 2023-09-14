import sys

sys.path.append('models')
import util.pointnetvlad_loss as PNV_loss
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def get_loss(args):
    if args["LOSS_FUNCTION"] == 'quadruplet':
        loss_function = PNV_loss.quadruplet_loss
    elif args["LOSS_FUNCTION"] == 'hphn_quadruplet':
        loss_function = PNV_loss.hphn_quadruplet_loss
    else:
        loss_function = PNV_loss.triplet_loss_wrapper
    return loss_function


def get_optimizer(args, net):
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    learning_rate = args["BASE_LEARNING_RATE"]
    if args["OPTIMIZER"] == 'momentum':
        optimizer = torch.optim.SGD(parameters, learning_rate,
                                    momentum=args["MOMENTUM"])
    elif args["OPTIMIZER"] == 'adam':
        optimizer = torch.optim.Adam(parameters, learning_rate)
    else:
        optimizer = None
        exit(0)

    if args["LEARNING_RATE_DECAY"] == 'step':
        lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.2)
    elif args["LEARNING_RATE_DECAY"] == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, args["MAX_EPOCH"],
                                         eta_min=learning_rate)
    else:
        lr_scheduler = None

    return optimizer, lr_scheduler
