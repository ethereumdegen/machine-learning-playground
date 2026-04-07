# Setting Up PyTorch From Scratch

## 1. Prerequisites

- **Python 3.8+** installed ([python.org](https://www.python.org/downloads/))
- **pip** (comes with Python) or **conda** (via Miniconda/Anaconda)
- A terminal / command line

Check your Python version:

```bash
python3 --version
```

---

## 2. Create a Virtual Environment

Always isolate your project dependencies.

```bash
# Create a virtual environment
python3 -m venv pytorch_env

# Activate it
# Linux/macOS:
source pytorch_env/bin/activate
# Windows:
pytorch_env\Scripts\activate
```

---

## 3. Install PyTorch

### Option A: pip (recommended for most users)

**CPU only:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**With NVIDIA GPU (CUDA 12.4):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Option B: conda

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

> Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to get the exact install command for your OS, package manager, and CUDA version.

---

## 4. Verify the Installation

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Quick tensor test
x = torch.tensor([1.0, 2.0, 3.0])
print(x * 2)  # tensor([2., 4., 6.])
```

Run it:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 5. Your First Neural Network

Create a file called `first_model.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# --- Step 1: Prepare dummy data ---
# XOR problem: 4 samples, 2 features
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# --- Step 2: Define a model ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

model = SimpleNet()

# --- Step 3: Set loss function and optimizer ---
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- Step 4: Train ---
for epoch in range(1000):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# --- Step 5: Test ---
with torch.no_grad():
    output = model(X)
    predicted = (output > 0.5).float()
    print(f"\nPredictions: {predicted.squeeze().tolist()}")
    print(f"Actual:      {y.squeeze().tolist()}")
```

Run it:

```bash
python first_model.py
```

---

## 6. Key Concepts Cheat Sheet

| Concept | What It Does |
|---|---|
| `torch.tensor()` | Creates a tensor (like a NumPy array, but GPU-capable) |
| `nn.Module` | Base class for all neural network models |
| `nn.Linear(in, out)` | A fully connected layer |
| `loss.backward()` | Computes gradients (backpropagation) |
| `optimizer.step()` | Updates model weights using gradients |
| `optimizer.zero_grad()` | Clears old gradients before each step |
| `torch.no_grad()` | Disables gradient tracking (used during inference) |
| `model.train()` | Sets model to training mode |
| `model.eval()` | Sets model to evaluation mode |

---

## 7. Common Next Steps

1. **Load real data** - Use `torch.utils.data.DataLoader` and `torchvision.datasets`
2. **Use GPU** - Move model and data with `.to('cuda')`
3. **Save/load models** - `torch.save()` and `torch.load()`
4. **TensorBoard** - Visualize training with `torch.utils.tensorboard`
5. **Pre-trained models** - Use `torchvision.models` (ResNet, etc.)

### Example: Using GPU

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)
X = X.to(device)
y = y.to(device)
```

### Example: Saving and Loading

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = SimpleNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

---

## 8. Useful Resources

- **Official Tutorials**: [pytorch.org/tutorials](https://pytorch.org/tutorials/)
- **API Docs**: [pytorch.org/docs/stable](https://pytorch.org/docs/stable/index.html)
- **PyTorch Forums**: [discuss.pytorch.org](https://discuss.pytorch.org/)
