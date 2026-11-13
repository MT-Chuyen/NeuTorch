# NeuTorch

Lightweight, zero-boilerplate Computer Vision training on top of PyTorch.

Train detection, segmentation, or classification models with a single function call — AMP, TensorBoard, callbacks, and torchmetrics evaluators included.

## Install

From GitHub:

```bash
pip install git+https://github.com/MT-Chuyen/NeuTorch.git
```

Editable local install (development):

```bash
git clone https://github.com/MT-Chuyen/NeuTorch.git
cd NeuTorch
pip install -e .
```

## Quickstart

```python
from neutorch import train_vision_task

history = train_vision_task(
    model,
    train_loader,
    val_loader,
    task="segmentation",  # "detection" | "segmentation" | "classification"
    epochs=50,
)
```

Or use the building blocks directly:

```python
from neutorch import Trainer, build_evaluator
from torch.optim import AdamW
import torch.nn as nn

optimizer = AdamW(model.parameters(), lr=1e-4)
evaluator = build_evaluator("classification", num_classes=10)

trainer = Trainer(
    model,
    optimizer,
    nn.CrossEntropyLoss(),
    task="classification",
    evaluator=evaluator,
    use_amp=True,
)
history = trainer.fit(train_loader, val_loader, epochs=20)
```

## Features

- **`train_vision_task()`** — one-call training for detection / segmentation / classification
- **`Trainer`** — AMP (`torch.cuda.amp`), TensorBoard logging, checkpoint-ready hooks
- **Callbacks** — `on_train_start`, `on_epoch_end`, `on_batch_end`, …
- **Evaluators** — GPU-resident mAP, mIoU, and accuracy via torchmetrics
- **Device helpers** — automatic CUDA / MPS / CPU selection

## Requirements

- Python 3.9+
- PyTorch 2.0+
- torchmetrics 1.0+
- tensorboard 2.14+

## License

MIT
