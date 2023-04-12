import timm
import torch
import torchvision.transforms as transforms
from utils import *

RANDOM_STATE = 777
torch.manual_seed(RANDOM_STATE)

IMAGE_SIZE = 224
DEVICE = 'cuda'

SAVE_PATH = 'checkpoints/resnet101_conv.pth'

LABEL_NAME = ['Wake', 'N1', 'N2', 'N3', 'REM']

test_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

model = timm.models.resnet101(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(
    7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(
    in_features=2048, out_features=len(LABEL_NAME), bias=True)
model = model.to(DEVICE)
model = nn.DataParallel(model)
