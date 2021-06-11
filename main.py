import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import copy

from . import Vgg16
from . import StyleTransfer

work_model = Vgg16()
