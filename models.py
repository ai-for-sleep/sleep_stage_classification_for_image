import torch
import torch.nn as nn
from torch.autograd import Variable


class SleepStager(nn.Module):
    def __init__(self, encoder, seq_len=5, num_classes=5):
        super().__init__()

        self.out_idx = seq_len // 2
        self.encoder = encoder
        self.num_layers = 3
        self.hidden_size = 512

        self.lstm = nn.LSTM(input_size=2048, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, bidirectional=True, bias=False)

        self.fc1 = nn.Linear(self.hidden_size*2, num_classes, bias=False)

    def init_hidden(self, batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = Variable(torch.zeros(self.num_layers*2,
                          batch_size, self.hidden_size).cuda())
        cell = Variable(torch.zeros(self.num_layers*2,
                        batch_size, self.hidden_size).cuda())

        return hidden, cell

    def forward(self, x):
        # b, seq_len, h, w
        seq = []

        for t in range(x.size(1)):
            with torch.no_grad():
                # b, 1, 224, 224
                x_t = x[:, t, :, :].unsqueeze(1)
                # b, 512
                x_t = self.encoder(x_t)

                seq.append(x_t)

        # b, seq_len, 512
        seq = torch.stack(seq, dim=1).squeeze(2)
        self.lstm.flatten_parameters()

        h0, c0 = self.init_hidden(x.size(0))
        out, hidden = self.lstm(seq, (h0, c0))

        # b, 512
        out = self.fc1(out[:, self.out_idx, :])

        # b, 5
        return out
