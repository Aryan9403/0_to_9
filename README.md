# MNIST CNN

A minimal convolutional neural network trained on MNIST in PyTorch.

## Files

| File | Purpose |
|------|---------|
| `model.py` | `MnistCNN` — two conv blocks + two FC layers |
| `data.py` | Downloads MNIST and returns train/test `DataLoader`s |
| `train.py` | Training loop, evaluation, entry point |

## Run locally

```bash
pip install torch torchvision
python train.py
```

## Run on Google Colab

1. Upload `model.py`, `data.py`, and `train.py` to the Colab session (or mount Drive).
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**.
3. In a cell:

```python
!python train.py
```

Expected test accuracy: ~99% after 5 epochs.

## Architecture

```
Input (1×28×28)
  → Conv2d(1→32, 3×3) + ReLU + MaxPool  →  32×14×14
  → Conv2d(32→64, 3×3) + ReLU + MaxPool →  64×7×7
  → Flatten                              →  3136
  → Linear(3136→128) + ReLU + Dropout
  → Linear(128→10)                       →  logits
```
