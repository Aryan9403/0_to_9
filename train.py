import torch
import torch.nn as nn
import torch.optim as optim

from model import MnistCNN
from data import get_loaders

NUM_EPOCHS = 5
LR         = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)            # forward pass
        loss   = criterion(logits, labels)

        optimizer.zero_grad()             # clear old gradients
        loss.backward()                   # backprop: compute dL/dw for every w
        optimizer.step()                  # weight update: w <- w - lr*grad

        running_loss += loss.item()
        if batch_idx % 200 == 0:
            print(f"    batch {batch_idx:4d}/{len(loader)} | loss: {loss.item():.4f}")

    return running_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def main():
    print(f"Using device: {device}\n")
    train_loader, test_loader = get_loaders()

    model     = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        avg_loss = train_epoch(model, train_loader, criterion, optimizer)
        accuracy = evaluate(model, test_loader)
        print(f"  avg loss: {avg_loss:.4f} | test acc: {accuracy * 100:.2f}%\n")

    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Model saved to mnist_cnn.pth")


if __name__ == "__main__":
    main()
