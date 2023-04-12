import json
import torch
import torch.nn as nn
import numpy as np
import timm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models import SleepStager
from utils import *

RANDOM_STATE = 777
torch.manual_seed(RANDOM_STATE)

CHANNELS = [
    (0, 0, 224, 26),
    (0, 26, 224, 62),
    (0, 62, 224, 81),
    (0, 81, 224, 89),
    (0, 89, 224, 99),
    (0, 99, 224, 123),
    (0, 123, 224, 133),
    (0, 133, 224, 150),
    (0, 150, 224, 160),
    (0, 160, 224, 188),
    (0, 188, 224, 224)
]

IMAGE_SIZE = 224
BATCH_SIZE = 1
LR_START = 0.1
EPOCHS = 10
DEVICE = 'cuda'
N_SPLIT = 5
SEQ_LEN = 5
NUM_WORKER = 8

DATA_PATH = 'data/PSG'
JSON_PATH = 'data/PSG.json'
CHECKPOINT_PATH = 'checkpoints'
LOG_PATH = 'lstm.log'

LABEL_NAME = ['Wake', 'N1', 'N2', 'N3', 'REM']

logger = get_logger(LOG_PATH, 'train')

test_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

with open(JSON_PATH, 'r') as f:
    json_data = json.load(f)

patients = json_data['Patient']

_, test_patients = train_test_split(
    patients, test_size=0.1, random_state=RANDOM_STATE)
test_patients = np.array(test_patients)

test_dataset = SleepSeqDataset(
    test_patients, DATA_PATH, LABEL_NAME, SEQ_LEN, test_transforms)
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKER,
                         pin_memory=True,
                         shuffle=False)

fold_labels = []
fold_predicts = []
fold_probs = []

for i in range(0, N_SPLIT):
    print(f'fold {i}')
    LOAD_PATH = f'resnet101_lstm_224_fold_{i}.pth'

    # encoder
    encoder = timm.models.resnet101(pretrained=False)
    encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(
        7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    encoder.fc = nn.Identity()

    # lstm
    model = SleepStager(encoder, SEQ_LEN, num_classes=len(LABEL_NAME))
    model = model.to(DEVICE)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(LOAD_PATH))
    model.eval()

    test_correct = 0
    test_loss = 0
    test_total = len(test_loader)

    total_labels = []
    total_predicted = []
    total_prob = []

    for (data, labels, number) in tqdm(test_loader, total=test_total):
        data = data.float().to(DEVICE)[0]
        labels = labels.long().to(DEVICE)[0]

        pred = model(data)
        _, predicted = torch.max(pred, 1)

        prob_array = pred.softmax(-1).detach().cpu().numpy()
        label_array = labels.detach().cpu().numpy()
        predicted_array = predicted.detach().cpu().numpy()

        total_labels.extend(list(label_array))
        total_predicted.extend(list(predicted_array))
        total_prob.extend(list(prob_array))

    fold_labels.append(total_labels)
    fold_predicts.append(total_predicted)
    fold_probs.append(total_prob)
