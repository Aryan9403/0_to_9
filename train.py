import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm

from model import MnistCNN
from data import get_loaders

NUM_EPOCHS = 5
LR         = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="  training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)            # forward pass
        loss   = criterion(logits, labels)

        optimizer.zero_grad()             # clear old gradients
        loss.backward()                   # backprop: compute dL/dw for every w
        optimizer.step()                  # weight update: w <- w - lr*grad

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

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


def plot_progress(losses, accuracies):
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(losses) + 1)

    ax1.plot(epochs, losses, "b-o", linewidth=2, markersize=6)
    ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in accuracies], "g-o", linewidth=2, markersize=6)
    ax2.set_title("Test Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim([90, 100])
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Epoch {len(losses)} / {NUM_EPOCHS}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def visual_test(model, loader):
    """
    Full test-set evaluation with two visual panels:
      1. Confusion matrix — which digits get confused with which.
      2. Misclassified examples — up to 32 images the model got wrong.
    """
    model.eval()

    all_preds, all_labels, wrong_images, wrong_preds, wrong_labels = [], [], [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  testing ", leave=False):
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            # collect misclassified images (up to 32 total)
            mask = preds != labels
            if mask.any() and len(wrong_images) < 32:
                wrong_images.append(images[mask].cpu())
                wrong_preds.append(preds[mask].cpu())
                wrong_labels.append(labels[mask].cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # ── Confusion matrix ──────────────────────────────────────────────────
    conf = torch.zeros(10, 10, dtype=torch.long)
    for t, p in zip(all_labels, all_preds):
        conf[t][p] += 1

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    im = axes[0].imshow(conf.numpy(), cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_xticks(range(10))
    axes[0].set_yticks(range(10))
    plt.colorbar(im, ax=axes[0])

    for r in range(10):
        for c in range(10):
            axes[0].text(c, r, conf[r, c].item(), ha="center", va="center",
                         fontsize=8, color="white" if conf[r, c] > conf.max() * 0.5 else "black")

    # ── Misclassified examples ─────────────────────────────────────────────
    wrong_images = torch.cat(wrong_images)[:32]
    wrong_preds  = torch.cat(wrong_preds)[:32]
    wrong_labels = torch.cat(wrong_labels)[:32]
    n            = len(wrong_images)
    cols         = 8
    rows         = (n + cols - 1) // cols

    inner = axes[1].inset_axes([0, 0, 1, 1])
    inner.set_visible(False)
    axes[1].axis("off")
    axes[1].set_title(f"Misclassified Examples ({n} shown)", fontsize=14, fontweight="bold")

    grid = fig.add_gridspec(rows, cols,
                            left=0.52, right=0.98, top=0.88, bottom=0.05,
                            wspace=0.1, hspace=0.6)

    for i in range(n):
        ax = fig.add_subplot(grid[i // cols, i % cols])
        ax.imshow(wrong_images[i].squeeze(), cmap="gray")
        ax.set_title(f"p:{wrong_preds[i].item()} t:{wrong_labels[i].item()}",
                     fontsize=7, color="red")
        ax.axis("off")

    accuracy = (all_preds == all_labels).float().mean().item()
    fig.suptitle(f"Test Results  |  accuracy: {accuracy * 100:.2f}%  |  errors: {(all_preds != all_labels).sum().item()} / {len(all_labels)}",
                 fontsize=15, fontweight="bold")
    plt.show()


def main():
    print(f"Using device: {device}\n")
    train_loader, test_loader = get_loaders()

    model     = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    losses, accuracies = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer)
        accuracy = evaluate(model, test_loader)
        losses.append(avg_loss)
        accuracies.append(accuracy)
        plot_progress(losses, accuracies)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | loss: {avg_loss:.4f} | test acc: {accuracy * 100:.2f}%")

    visual_test(model, test_loader)

    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("\nModel saved to mnist_cnn.pth")


if __name__ == "__main__":
    main()
