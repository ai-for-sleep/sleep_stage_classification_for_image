import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


channels = [
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


def get_logger(log_path, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.WARNING)
    logger.addHandler(streamHandler)

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)

    return logger


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_path=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def train_conv(model, train_loader, optimizer, criterion, device="cuda"):
    model.train()

    total = len(train_loader)
    train_correct = 0
    train_loss = 0

    for i, (data, labels) in enumerate(tqdm(train_loader, total=total)):
        optimizer.zero_grad()

        data = data.float().to(device)
        labels = labels.long().to(device)

        pred = model(data)
        _, predicted = torch.max(pred, 1)
        train_correct += (predicted == labels).sum().item()
        loss = criterion(pred, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(
                f"Train Acc : {train_correct / (len(data) * (i + 1))} || Train Loss : {train_loss / (i + 1)}")

    return train_correct, train_loss


def train_conv_linemix(model, train_loader, optimizer, criterion, device="cuda"):
    model.train()

    total = len(train_loader)
    train_correct = 0
    train_loss = 0

    for i, (data, labels) in enumerate(tqdm(train_loader, total=total)):
        optimizer.zero_grad()

        # generate mixed sample
        rand_index = torch.randperm(data.size()[0]).cuda()

        data = data.to(device)
        labels_a = labels.to(device)
        labels_b = labels[rand_index].to(device)

        # select channel
        c = np.random.randint(len(channels))
        pick = channels[c]
        y_min, y_max = pick[1], pick[3]
        data[:, :, y_min:y_max, :] = data[rand_index, :, y_min:y_max, :]

        # adjust lambda to exactly match pixel ratio
        lam = (y_max - y_min) / 224

        # compute output
        pred = model(data)

        _, predicted = torch.max(pred, 1)
        train_correct += (predicted == labels_a).sum().item()
        loss = criterion(pred, labels_a) * (1. - lam) + \
            criterion(pred, labels_b) * lam

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(
                f"Train Acc : {train_correct / (len(data) * (i + 1))} || Train Loss : {train_loss / (i + 1)}")

    return train_correct, train_loss


def valid_conv(model, valid_loader, criterion, device="cuda"):
    model.eval()

    total = len(valid_loader)
    valid_correct = 0
    valid_loss = 0

    with torch.no_grad():
        for (data, labels) in tqdm(valid_loader, total=total):
            data = data.float().to(device)
            labels = labels.long().to(device)

            pred = model(data)
            _, predicted = torch.max(pred, 1)
            valid_correct += (predicted == labels).sum().item()
            loss = criterion(pred, labels)

            valid_loss += loss.item()

    return valid_correct, valid_loss


def train_lstm(model, train_loader, optimizer, criterion, scaler=None, device="cuda"):
    model.train()

    total = len(train_loader)
    num_of_data = 0
    train_correct = 0
    train_loss = 0

    for data, labels, number in tqdm(train_loader, total=total):
        optimizer.zero_grad()

        data = data.float().to(device)[0]
        labels = labels.long().to(device)[0]

        if scaler is None:
            # 1, b, num_classes
            pred = model(data)
            # 1, b
            _, predicted = torch.max(pred, 1)

            correct = (predicted == labels).sum().item()
            correct_ratio = correct / len(labels)

            train_correct += correct
            num_of_data += len(labels)

            # | b, num_classes | b |
            loss = criterion(pred, labels)
            train_loss += loss.item() / len(labels)

            loss.backward()
            optimizer.step()

        else:
            with torch.cuda.amp.autocast():
                # 1, b, num_classes
                pred = model(data)
                # 1, b
                _, predicted = torch.max(pred, 1)

                correct = (predicted == labels).sum().item()
                correct_ratio = correct / len(labels)

                train_correct += correct
                num_of_data += len(labels)

                # | b, num_classes | b |
                loss = criterion(pred, labels)
                train_loss += loss.item() / len(labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        print(
            f"[TRAIN / {number}] [{len(predicted)} / {correct}] [RATIO / {correct_ratio}]")

    train_acc = train_correct / num_of_data

    return train_acc, train_loss


def valid_lstm(model, valid_loader, criterion, device="cuda"):
    model.eval()

    total = len(valid_loader)
    num_of_data = 0
    valid_correct = 0
    valid_loss = 0

    with torch.no_grad():
        for data, labels, number in tqdm(valid_loader, total=total):
            data = data.float().to(device)[0]  # (1, batch) -> (batch)
            labels = labels.long().to(device)[0]

            pred = model(data)
            _, predicted = torch.max(pred, 1)

            correct = (predicted == labels).sum().item()
            correct_ratio = correct / len(labels)

            valid_correct += correct
            num_of_data += len(labels)

            print(
                f"[VALID / {number}] [{len(predicted)} / {correct}] [RATIO / {correct_ratio}]")

            loss = criterion(pred, labels)
            valid_loss += loss.item() / len(labels)

    valid_acc = valid_correct / num_of_data

    return valid_acc, valid_loss


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='model.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


def lineout(mask_size, p, mask_color=0):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        cymin, cymax = 0, h + offset

        cy = np.random.randint(cymin, cymax)
        ymin = cy - mask_size_half
        ymax = ymin + mask_size
        ymin = max(0, ymin)
        ymax = min(h, ymax)
        image[ymin:ymax, 0:w] = mask_color

        return image

    return _cutout


class SleepConvDataset(Dataset):
    def __init__(self, patients, data_path, label_name, transforms=None):
        self.patients = patients
        self.data_path = data_path
        self.label_name = label_name
        self.transforms = transforms

        self.total_paths = []
        self.total_labels = []
        self.total_numbers = []

        for patient in self.patients:
            start = True

            paths = []
            labels = []

            patient_number = patient['Patient_Number']
            events = patient['Event']
            branch = '-'.join(patient_number.split('-')[:-1])
            path = self.data_path / branch / \
                patient_number / f"{patient_number}_standard"

            if 'NX' in patient_number:
                for n in range(0, len(events)):
                    cur_event = events[n]

                    stage = cur_event['Event_Label']

                    if stage == 'R':
                        stage = 'REM'

                    if stage in self.label_name:
                        if start:
                            start = False

                            init_start_epoch = int(cur_event['Start_Epoch'])
                            init_end_epoch = int(cur_event['End_Epoch'])

                            for epoch in range(init_start_epoch, init_end_epoch):
                                image_number = str(epoch).zfill(4)

                                label = self.label_name.index(stage)
                                image_path = path / \
                                    f"{patient_number}_{image_number}.png"

                                if os.path.exists(image_path):
                                    paths.append(image_path)
                                    labels.append(label)

                        else:
                            start_epoch = int(cur_event['Start_Epoch'])
                            end_epoch = int(cur_event['End_Epoch'])

                            if init_end_epoch <= start_epoch:
                                if not init_end_epoch == start_epoch:
                                    for epoch in range(init_end_epoch, start_epoch):
                                        image_number = str(epoch).zfill(4)

                                        label = self.label_name.index("Wake")
                                        image_path = path / \
                                            f"{patient_number}_{image_number}.png"

                                        if os.path.exists(image_path):
                                            paths.append(image_path)
                                            labels.append(label)

                                for epoch in range(start_epoch, end_epoch):
                                    image_number = str(epoch).zfill(4)

                                    label = self.label_name.index(stage)
                                    image_path = path / \
                                        f"{patient_number}_{image_number}.png"

                                    if os.path.exists(image_path):
                                        paths.append(image_path)
                                        labels.append(label)

                                init_end_epoch = end_epoch

                if len(paths) == 0:
                    print(patient_number)

                else:
                    self.total_paths += paths
                    self.total_labels += labels
                    self.total_numbers += patient_number

    def __len__(self):
        return len(self.total_labels)

    def __getitem__(self, idx):
        img = Image.open(self.total_paths[idx])
        label = self.total_labels[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class SleepSeqDataset(Dataset):
    def __init__(self, patients, root_path, label_name, seq_len=5, transforms=None):
        self.total_paths = []
        self.total_labels = []
        self.total_number = []
        self.seq_len = seq_len
        self.pad_len = seq_len // 2
        self.sleep_label_names = label_name

        for patient in patients:
            paths = []
            labels = []

            start = True

            patient_number = patient['Patient_Number']
            events = patient['Event']

            branch = '-'.join(patient_number.split('-')[:-1])
            image_folder_path = root_path / branch / \
                patient_number / f"{patient_number}_standard"

            for i in range(0, len(events)):
                stage = events[i]['Event_Label']

                if stage == 'R':
                    stage = 'REM'

                if stage not in self.sleep_label_names:
                    continue

                if start:
                    start = False

                    init_start_epoch = int(events[i]['Start_Epoch'])
                    init_end_epoch = int(events[i]['End_Epoch'])

                    for epoch in range(init_start_epoch, init_end_epoch):
                        image_number = str(epoch).zfill(4)

                        label = self.sleep_label_names.index(stage)
                        image_path = image_folder_path / \
                            f"{patient_number}_{image_number}.png"

                        if os.path.exists(image_path):
                            paths.append(image_path)
                            labels.append(label)

                else:
                    start_epoch = int(events[i]['Start_Epoch'])
                    end_epoch = int(events[i]['End_Epoch'])

                    if init_end_epoch <= start_epoch:
                        if init_end_epoch < start_epoch:
                            for epoch in range(init_end_epoch, start_epoch):
                                image_number = str(epoch).zfill(4)

                                label = self.sleep_label_names.index("Wake")
                                image_path = image_folder_path / \
                                    f"{patient_number}_{image_number}.png"

                                if os.path.exists(image_path):
                                    paths.append(image_path)
                                    labels.append(label)

                        for epoch in range(start_epoch, end_epoch):
                            image_number = str(epoch).zfill(4)

                            label = self.sleep_label_names.index(stage)
                            image_path = image_folder_path / \
                                f"{patient_number}_{image_number}.png"

                            if os.path.exists(image_path):
                                paths.append(image_path)
                                labels.append(label)

                        init_end_epoch = end_epoch

            if len(paths) > 0:
                self.total_paths.append(paths)
                self.total_labels.append(labels)
                self.total_number.append(patient_number)

        self.transforms = transforms

    def __len__(self):
        return len(self.total_paths)

    def __getitem__(self, idx):
        paths = self.total_paths[idx]
        labels = self.total_labels[idx]
        labels = np.array(labels)

        number = self.total_number[idx]

        start_pad = [paths[0] for _ in range(self.pad_len)]
        end_pad = [paths[-1] for _ in range(self.pad_len)]

        paths = np.append(start_pad, paths)
        paths = np.append(paths, end_pad)

        seq = []

        for p in paths:
            img = Image.open(p)

            if self.transforms is not None:
                img = self.transforms(img)

            seq.append(img)

        seq = torch.stack(seq, dim=1).squeeze(0)
        seq = seq.unfold(0, self.seq_len, 1).permute(0, 3, 1, 2)

        return (seq, labels, number)
