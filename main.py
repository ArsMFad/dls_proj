import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import copy

import Vgg16
import StyleTransfer as s_trans

from PIL import Image
import PIL


def return_result(cont_img_path, style_img_path, usr_name, quality, model_choose):
    device = torch.device("cpu")
    alexnet_model = models.alexnet(pretrained=True)
    num_ftrs = alexnet_model.classifier[6].in_features
    alexnet_model.classifier[6] = nn.Linear(num_ftrs, 2)
    alexnet_model.load_state_dict(torch.load('modelka_alexnet', map_location=device))
    alexnet_model = alexnet_model.features.to(device).eval()

    vgg_model = models.vgg19_bn(pretrained=True)
    num_ftrs = vgg_model.classifier[6].in_features
    vgg_model.classifier[6] = nn.Linear(num_ftrs, 2)
    vgg_model.load_state_dict(torch.load('modelka_vgg', map_location=device))
    vgg_model = vgg_model.features.to(device).eval()

    if model_choose == "0":
        work_model = alexnet_model
    else:
        work_model = vgg_model

    content_img = s_trans.image_loader(cont_img_path)
    style_img = s_trans.image_loader(style_img_path)

    quality = int(quality)
    if model_choose == "0":
        quality *= 10
        if quality < 0 or quality > 30:
            quality = 20
    else:
        if quality < 0 or quality > 3:
            quality = 2
    quality *= 100

    output = s_trans.run_style_transfer(work_model, s_trans.cnn_normalization_mean,
                                        s_trans.cnn_normalization_std, content_img, style_img, content_img, quality)
    unloader = transforms.ToPILImage()
    output = output.cpu().clone()
    output = output.squeeze(0)
    output = unloader(output)
    output = output.save(str(usr_name) + "result.jpg")

#return_result("cont339492786.jpg", "style339492786.jpg", "alexei", 1)
