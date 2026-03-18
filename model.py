import torch.nn as nn


class MnistCNN(nn.Module):
    # Input: (B, 1, 28, 28)  ->  conv+pool twice  ->  FC  ->  10 logits
    def __init__(self):
        super().__init__()
        # Each Conv2d learns N filters that slide across the input feature maps.
        # padding=1 preserves spatial size; MaxPool2d halves it: 28->14->7
        self.conv1   = nn.Conv2d(1,  32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.fc2     = nn.Linear(128, 10)          # raw logits, no softmax
        self.relu    = nn.ReLU()                   # max(0, x) -- breaks linearity
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))    # (B, 32, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))    # (B, 64,  7,  7)
        x = x.view(x.size(0), -1)                 # (B, 3136)
        x = self.dropout(self.relu(self.fc1(x)))   # (B, 128)
        return self.fc2(x)                         # (B, 10)
