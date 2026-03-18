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
pip install -r requirements.txt
python train.py
```

## Run on Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

2. Set the runtime to GPU: **Runtime → Change runtime type → T4 GPU**

3. Clone this repo:

```python
!git clone https://github.com/your-username/mnist-cnn.git
%cd mnist-cnn
```

4. Install dependencies:

```python
!pip install -r requirements.txt -q
```

5. Run training:

```python
!python train.py
```

6. Download the saved model — Colab runtimes are temporary, so save `mnist_cnn.pth` before the session ends:

```python
from google.colab import files
files.download("mnist_cnn.pth")
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
