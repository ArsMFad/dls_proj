import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import copy

import Vgg16
import matplotlib.pyplot as plt
import StyleTransfer as s_trans

from PIL import Image
import PIL

#work_model = Vgg16()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
work_model = models.vgg19(pretrained=True).features.to(device).eval()

content_img = s_trans.image_loader("grusha.jpg")
style_img = s_trans.image_loader("picasso.jpg")

input_img = content_img.clone()
output = s_trans.run_style_transfer(work_model, s_trans.cnn_normalization_mean, s_trans.cnn_normalization_std, content_img, style_img, content_img)
unloader = transforms.ToPILImage()
output = output.cpu().clone()
output = output.squeeze(0)
output = unloader(output)
output = output.save("result.jpg")
