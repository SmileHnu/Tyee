## Tyee.criterion Module

The `Tyee.criterion` module extends PyTorch's existing loss functions (`torch.nn.Module`) by providing additional, custom-implemented loss functions commonly used in various deep learning tasks. You can seamlessly use loss functions provided by PyTorch or opt for the specialized loss functions implemented within Tyee.

Below is a summary of the custom loss functions available in `Tyee.criterion`:

**Custom Loss Functions Table**

| Class Name                   | Functional Description                                       |
| ---------------------------- | ------------------------------------------------------------ |
| [`LabelSmoothingCrossEntropy`](#1-labelsmoothingcrossentropy) | Implements cross-entropy loss with label smoothing to regularize the model and prevent overconfidence. |
| [`SoftTargetCrossEntropy`](#2-softtargetcrossentropy)     | Computes cross-entropy loss when targets are soft probability distributions instead of hard labels. |
| [`FocalLoss`](#3-focalloss)                  | Implements Focal Loss, which addresses class imbalance by down-weighting well-classified examples. |
| [`BinnedRegressionLoss`](#4-binnedregressionloss)       | Converts continuous regression targets into binned probability distributions and computes cross-entropy against predictions. |

------

### Detailed Descriptions

Below are detailed descriptions of each custom loss function, including their initialization parameters and usage.

#### 1. LabelSmoothingCrossEntropy

Implements the cross-entropy loss with label smoothing. Label smoothing is a regularization technique that prevents the model from becoming too confident about its predictions.

**Initialization:**

```python
LabelSmoothingCrossEntropy(smoothing=0.1)
```

- **`smoothing`** (`float`, optional): The label smoothing factor. Must be less than 1.0. A common value is 0.1. (Default: `0.1`)

**Usage:**

Calculates the loss between model predictions (`x`) and true targets (`target`).

```python
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
loss = criterion(x, target)
```

- **`x`** (`torch.Tensor`): The model's raw output logits (before softmax). Shape: `(N, C)` where `N` is the batch size and `C` is the number of classes.
- **`target`** (`torch.Tensor`): The ground truth labels (class indices). Shape: `(N)`.
- **Returns** (`torch.Tensor`): A scalar tensor representing the mean loss.

------

#### 2. SoftTargetCrossEntropy

Implements cross-entropy loss for soft targets. This is useful when the ground truth labels are not hard class indices but rather probability distributions over classes (e.g., from knowledge distillation or when targets have inherent uncertainty).

**Initialization:**

```python
SoftTargetCrossEntropy()
```

This loss function does not require any specific initialization parameters beyond those of `nn.Module`.

**Usage:**

Calculates the loss between model predictions (`x`) and soft targets (`target`).

```python
criterion = SoftTargetCrossEntropy()
loss = criterion(x, target)
```

- **`x`** (`torch.Tensor`): The model's raw output logits (before softmax). Shape: `(N, C)`.
- **`target`** (`torch.Tensor`): The ground truth soft labels (probability distributions). Shape: `(N, C)`. Each row should sum to 1.
- **Returns** (`torch.Tensor`): A scalar tensor representing the mean loss.

------

#### 3. FocalLoss

Implements Focal Loss, which is particularly useful for training models on datasets with a severe class imbalance. It reshapes the standard cross-entropy loss to down-weight the loss assigned to well-classified examples, focusing more on hard, misclassified examples.

**Initialization:**

```python
FocalLoss(gamma=2, alpha=None)
```

- **`gamma`** (`float`, optional): The focusing parameter. Higher values of gamma apply a stronger down-weighting to easy examples. (Default: `2`)
- **`alpha`** (`torch.Tensor` or `float`, optional): A weighting factor for each class. If a `float`, it is typically applied to the positive class in binary classification. If a `torch.Tensor`, it should have `C` elements, where `C` is the number of classes, providing a weight for each class. (Default: `None`)

**Usage:**

Calculates the Focal Loss between model predictions (`logits`) and true targets (`targets`).

```python
# Example for multi-class with alpha
# alpha_tensor = torch.tensor([0.25, 0.25, 0.25, 0.25]) # if 4 classes
# criterion = FocalLoss(gamma=2, alpha=alpha_tensor)

criterion = FocalLoss(gamma=2)
loss = criterion(logits, targets)
```

- **`logits`** (`torch.Tensor`): The model's raw output logits. Shape: `(N, C)`.
- **`targets`** (`torch.Tensor`): The ground truth labels (class indices). Shape: `(N)`.
- **Returns** (`torch.Tensor`): A scalar tensor representing the mean loss.

------

#### 4. BinnedRegressionLoss

Implements a loss function for regression tasks where the continuous target variable is discretized into bins. The true continuous value is converted into a soft probability distribution over these bins (typically using a Gaussian), and the loss is computed as the cross-entropy between the model's predicted distribution and this target distribution. This is often used in tasks like estimating physiological signals (e.g., heart rate) where predicting a distribution can be more robust.

**Initialization:**

```python
BinnedRegressionLoss(dim, min_hz, max_hz, sigma_y)
```

- **`dim`** (`int`): The number of bins to discretize the target range into.
- **`min_hz`** (`float`): The minimum value of the predictable range in Hertz (Hz). This will be converted to Beats Per Minute (BPM) internally to define the lower edge of the first bin.
- **`max_hz`** (`float`): The maximum value of the predictable range in Hertz (Hz). This will be converted to BPM internally to define the upper edge of the last bin.
- **`sigma_y`** (`float`): The standard deviation (in BPM) for the Gaussian distribution used to convert the continuous true target value into a soft distribution over the bins. This controls the "softness" or spread of the target distribution.

**Usage:**

Calculates the binned regression loss.

```python
criterion = BinnedRegressionLoss(dim=100, min_hz=0.5, max_hz=3.0, sigma_y=2.5)
loss = criterion(y_pred, y_true)
```

- **`y_pred`** (`torch.Tensor`): The model's predicted probability distribution over the bins. Expected to be outputs from a softmax layer. Shape: `(batch_size, seq_len, dim)`.
- **`y_true`** (`torch.Tensor`): The ground truth continuous target values (e.g., heart rate in BPM). Shape: `(batch_size, seq_len)`.
- **Returns** (`torch.Tensor`): A scalar tensor representing the mean loss.