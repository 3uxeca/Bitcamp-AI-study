import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

# 데이터 전처리 방식 지정
transform = transforms.Compose([
    transforms.ToTensor(),
    # 데이터를 pytorch의 Tensor 방식으로 바꾼다.
    transforms.Normalize(mean=(0.5,), std=(0.5,))
    # 픽셀값 0 ~ 1 -> -1 ~ 1
])

