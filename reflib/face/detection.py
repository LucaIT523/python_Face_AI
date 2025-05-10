import os
import torch
from torch import nn
from copy import deepcopy
from reflib.face.detlib.retinaface.retinaface import RetinaFace


def init_detection_model( model_name, half=False, device='cuda'):
    if 'retinaface' in model_name:
        model = init_retinaface_model(model_name, half, device)
    # elif 'YOLOv5' in model_name:
        # model = init_yolov5face_model(model_name, device)
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    return model


def init_retinaface_model( model_name, half=False, device='cuda'):

    model = RetinaFace(network_name='resnet50', half=half)
    model_path = 'reflib/model/Resnet50.pth'
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)

    return model

