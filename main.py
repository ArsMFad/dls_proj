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


#work_model = Vgg16()
async def return_result(cont_img_path, style_img_path, usr_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_model = models.vgg19(pretrained=True).features.to(device).eval()

    content_img = s_trans.image_loader(cont_img_path)
    style_img = s_trans.image_loader(style_img_path)

    output = s_trans.run_style_transfer(work_model, s_trans.cnn_normalization_mean,
                                        s_trans.cnn_normalization_std, content_img, style_img, content_img)
    unloader = transforms.ToPILImage()
    output = output.cpu().clone()
    output = output.squeeze(0)
    output = unloader(output)
    output = output.save(str(usr_name) + "result.jpg")
