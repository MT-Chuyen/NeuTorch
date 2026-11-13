"""Core training loop with AMP and TensorBoard support."""

from __future__ import annotations

from typing import Any, Literal, Mapping, MutableMapping, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from neutorch.engine.callbacks import Callback
from neutorch.metrics.cv_metrics import _BaseEvaluator, build_evaluator
from neutorch.utils.device import batch_to_device, get_device, module_to_device, to_device

VisionTask = Literal["detection", "segmentation", "classification"]


class Trainer:
    """Lightweight training orchestrator for Computer Vision models.

    Handles the core train/eval loop, automatic mixed precision (AMP),
    TensorBoard logging, and callback dispatch. Intended as a
    zero-boilerplate execution layer on top of PyTorch.

    Args:
        model: The ``nn.Module`` to train.
        optimizer: Optimizer instance (or factory result).
        loss_fn: Criterion mapping ``(predictions, targets) -> loss``.
            Ignored for ``task="detection"`` when the model returns a
            loss dict (e.g. torchvision detectors).
        device: Target device string or ``torch.device``. Resolved
            automatically when omitted.
        callbacks: Optional sequence of :class:`~neutorch.engine.callbacks.Callback`
            instances hooked into the training lifecycle.
        use_amp: Enable CUDA automatic mixed precision when ``True``.
        log_dir: Directory for TensorBoard event files. Logging is
            skipped when ``None``.
        grad_clip_norm: Optional max L2 norm for gradient clipping.
        task: Vision task flavour controlling the train/eval forward path.
        evaluator: Optional epoch-level metric accumulator
            (mAP / IoU / accuracy).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        *,
        device: Optional[Union[str, torch.device]] = None,
        callbacks: Optional[Sequence[Callback]] = None,
        use_amp: bool = False,
        log_dir: Optional[str] = None,
        grad_clip_norm: Optional[float] = None,
        task: VisionTask = "classification",
        evaluator: Optional[_BaseEvaluator] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device: torch.device = (
            torch.device(device) if device is not None else get_device()
        )
        self.callbacks: list[Callback] = list(callbacks) if callbacks else []
        self.use_amp = use_amp
        self.log_dir = log_dir
        self.grad_clip_norm = grad_clip_norm
        self.task: VisionTask = task  # type: ignore[assignment]
        self.evaluator = evaluator

        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.writer: Optional[SummaryWriter] = None
        self.current_epoch: int = 0
        self.global_step: int = 0
        self._amp_enabled: bool = False

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        *,
        epochs: int = 1,
        start_epoch: int = 0,
    ) -> Mapping[str, Any]:
        """Run the full training loop for ``epochs`` epochs.

        Args:
            train_loader: DataLoader yielding training batches.
            val_loader: Optional validation DataLoader evaluated at
                the end of each epoch.
            epochs: Number of epochs to train.
            start_epoch: Epoch index to resume from (inclusive).

        Returns:
            Aggregated training history (losses, metrics, etc.).
        """
        self.model, self.device = module_to_device(self.model, self.device)
        self.loss_fn = self.loss_fn.to(self.device)

        self._setup_amp()
        self._setup_logging()

        if self.evaluator is not None:
            self.evaluator.to(self.device)
            if self.evaluator.writer is None:
                self.evaluator.writer = self.writer

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        self._dispatch("on_train_start")

        end_epoch = start_epoch + epochs
        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch
            self._dispatch("on_epoch_start", epoch=epoch)

            train_metrics = self._train_epoch(train_loader)
            history["train_loss"].append(train_metrics.get("loss", float("nan")))

            epoch_metrics: dict[str, float] = {
                f"train_{key}": value for key, value in train_metrics.items()
            }

            if val_loader is not None:
                self._dispatch("on_validation_start")
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics.get("loss", float("nan")))
                for key, value in val_metrics.items():
                    if key == "loss":
                        continue
                    history.setdefault(f"val_{key}", []).append(value)
                epoch_metrics.update(
                    {f"val_{key}": value for key, value in val_metrics.items()}
                )
                self._dispatch("on_validation_end", metrics=val_metrics)

            self._dispatch("on_epoch_end", epoch=epoch, metrics=epoch_metrics)

        self._dispatch("on_train_end")

        if self.writer is not None:
            self.writer.flush()

        return history

    def train_epoch(self, train_loader: DataLoader) -> Mapping[str, float]:
        """Execute a single training epoch."""
        return self._train_epoch(train_loader)

    def _train_epoch(self, train_loader: DataLoader) -> Mapping[str, float]:
        """Run one training epoch with AMP-aware optimization."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            self._dispatch("on_batch_start", batch=batch, batch_idx=batch_idx)

            batch = batch_to_device(batch, self.device, non_blocking=True)
            inputs, targets = self._unpack_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self._amp_enabled):
                loss = self._forward_loss(inputs, targets)

            if self.scaler is not None and self._amp_enabled:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                self.optimizer.step()

            loss_value = float(loss.detach().item())
            running_loss += loss_value
            num_batches += 1
            self.global_step += 1

            step_outputs = {"loss": loss_value}

            if self.global_step % 50 == 0:
                self._log_scalars(
                    {"loss": loss_value},
                    step=self.global_step,
                    prefix="train/",
                )

            self._dispatch(
                "on_batch_end",
                batch=batch,
                batch_idx=batch_idx,
                outputs=step_outputs,
            )

        mean_loss = running_loss / max(num_batches, 1)
        return {"loss": mean_loss}

    def train_step(
        self,
        batch: Any,
    ) -> Mapping[str, float]:
        """Perform one optimization step on a mini-batch."""
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Mapping[str, float]:
        """Run evaluation over ``data_loader`` without gradient tracking.

        Accumulates ``self.evaluator`` metrics on-device across the loader,
        then computes / logs / resets them at the end of the pass.
        """
        self.model.eval()
        if self.evaluator is not None:
            self.evaluator.to(self.device)
            self.evaluator.reset()

        running_loss = 0.0
        num_batches = 0

        for batch in data_loader:
            batch = batch_to_device(batch, self.device, non_blocking=True)
            inputs, targets = self._unpack_batch(batch)

            with torch.cuda.amp.autocast(enabled=self._amp_enabled):
                outputs, loss = self._forward_eval(inputs, targets)

            if loss is not None:
                running_loss += float(loss.detach().item())
            num_batches += 1

            if self.evaluator is not None:
                self.evaluator.update(outputs, targets)

        metrics: dict[str, float] = {
            "loss": running_loss / max(num_batches, 1) if num_batches else float("nan"),
        }

        if self.evaluator is not None:
            metrics.update(
                self.evaluator.compute(log=True, step=self.global_step, reset=True)
            )

        self._log_scalars(metrics, step=self.global_step, prefix="val/")
        return metrics

    def _forward_loss(self, inputs: Any, targets: Any) -> Tensor:
        """Task-aware training forward that returns a scalar loss."""
        if self.task == "detection":
            loss_dict = self.model(inputs, targets)
            if not isinstance(loss_dict, Mapping):
                raise TypeError(
                    "Detection models must return a loss dict in train mode "
                    f"(got {type(loss_dict)!r})."
                )
            return sum(loss_dict.values())  # type: ignore[return-value]
        outputs = self.model(inputs)
        return self.loss_fn(outputs, targets)

    def _forward_eval(
        self, inputs: Any, targets: Any
    ) -> tuple[Any, Optional[Tensor]]:
        """Task-aware eval forward returning ``(outputs, optional_loss)``."""
        if self.task == "detection":
            outputs = self.model(inputs)
            return outputs, None
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        return outputs, loss

    def _setup_amp(self) -> None:
        """Initialize GradScaler / autocast state when AMP is enabled."""
        self._amp_enabled = bool(
            self.use_amp and self.device.type == "cuda" and torch.cuda.is_available()
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)

    def _setup_logging(self) -> None:
        """Create the TensorBoard ``SummaryWriter`` if ``log_dir`` is set."""
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def _log_scalars(
        self,
        metrics: Mapping[str, float],
        *,
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """Write scalar metrics to TensorBoard."""
        if self.writer is None:
            return
        log_step = self.global_step if step is None else step
        for name, value in metrics.items():
            self.writer.add_scalar(f"{prefix}{name}", value, log_step)

    def _dispatch(
        self,
        event: str,
        **kwargs: Any,
    ) -> None:
        """Invoke ``event`` on every registered callback."""
        for callback in self.callbacks:
            handler = getattr(callback, event, None)
            if handler is not None:
                handler(self, **kwargs)

    def _unpack_batch(self, batch: Any) -> tuple[Any, Any]:
        """Split a batch into ``(inputs, targets)``."""
        if isinstance(batch, Mapping):
            input_keys = ("images", "image", "inputs", "input", "x")
            target_keys = ("targets", "target", "labels", "label", "y")
            inputs = next((batch[k] for k in input_keys if k in batch), None)
            targets = next((batch[k] for k in target_keys if k in batch), None)
            if inputs is None or targets is None:
                raise ValueError(
                    "Dict batch must contain input and target keys "
                    f"(tried inputs={input_keys}, targets={target_keys})."
                )
            return inputs, targets

        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]

        raise TypeError(
            "Expected batch as (inputs, targets) or a dict with image/target keys, "
            f"got {type(batch)!r}."
        )

    def add_callback(self, callback: Callback) -> None:
        """Register an additional callback."""
        self.callbacks.append(callback)

    def save_checkpoint(
        self,
        path: str,
        *,
        extra: Optional[MutableMapping[str, Any]] = None,
    ) -> None:
        """Persist model, optimizer, scaler, and training state."""
        raise NotImplementedError

    def load_checkpoint(
        self,
        path: str,
        *,
        strict: bool = True,
    ) -> Mapping[str, Any]:
        """Restore model, optimizer, scaler, and training state."""
        raise NotImplementedError

    def to_device(self, batch: Any) -> Any:
        """Recursively move a batch to ``self.device``."""
        return to_device(batch, self.device, non_blocking=True)


def _infer_num_classes(model: nn.Module) -> Optional[int]:
    """Best-effort ``num_classes`` discovery from common CV model layouts."""
    if hasattr(model, "num_classes"):
        value = getattr(model, "num_classes")
        if isinstance(value, int) and value > 0:
            return value

    roi_heads = getattr(model, "roi_heads", None)
    box_predictor = getattr(roi_heads, "box_predictor", None) if roi_heads else None
    cls_score = getattr(box_predictor, "cls_score", None) if box_predictor else None
    if isinstance(cls_score, nn.Linear):
        return int(cls_score.out_features)

    last_linear: Optional[nn.Linear] = None
    last_conv: Optional[nn.Conv2d] = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
        elif isinstance(module, nn.Conv2d):
            last_conv = module
    if last_linear is not None:
        return int(last_linear.out_features)
    if last_conv is not None:
        return int(last_conv.out_channels)
    return None


def _default_loss(task: VisionTask) -> nn.Module:
    """Return a sensible default criterion for ``task``."""
    if task == "detection":
        # Unused when the detector returns its own loss dict; kept so
        # Trainer always has a module to ``.to(device)``.
        return nn.Identity()
    return nn.CrossEntropyLoss()


def train_vision_task(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    task: VisionTask,
    epochs: int,
    *,
    num_classes: Optional[int] = None,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    use_amp: bool = True,
    log_dir: Optional[str] = "runs",
    device: Optional[Union[str, torch.device]] = None,
    loss_fn: Optional[nn.Module] = None,
    **evaluator_kwargs: Any,
) -> Mapping[str, Any]:
    """Zero-boilerplate entry point for common vision training jobs.

    Instantiates AdamW, the task-appropriate evaluator, a :class:`Trainer`,
    and runs :meth:`Trainer.fit`.

    Example::

        history = train_vision_task(
            model, train_loader, val_loader, task="segmentation", epochs=50
        )

    Args:
        model: Model to train.
        train_dataloader: Training data.
        val_dataloader: Validation data (metrics computed each epoch).
        task: ``"detection"``, ``"segmentation"``, or ``"classification"``.
        epochs: Number of training epochs.
        num_classes: Required for segmentation / classification when it
            cannot be inferred from ``model``.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        use_amp: Enable CUDA AMP when available.
        log_dir: TensorBoard directory (``None`` disables logging).
        device: Optional device override.
        loss_fn: Optional criterion override.
        **evaluator_kwargs: Forwarded to :func:`build_evaluator`.

    Returns:
        Training history from :meth:`Trainer.fit`.
    """
    normalized: VisionTask = task.strip().lower()  # type: ignore[assignment]
    if normalized not in ("detection", "segmentation", "classification"):
        raise ValueError(
            f"Unknown task {task!r}. Expected 'detection', 'segmentation', "
            "or 'classification'."
        )

    resolved_classes = num_classes if num_classes is not None else _infer_num_classes(model)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = loss_fn if loss_fn is not None else _default_loss(normalized)
    evaluator = build_evaluator(
        normalized,
        num_classes=resolved_classes,
        device=device,
        **evaluator_kwargs,
    )

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device=device,
        use_amp=use_amp,
        log_dir=log_dir,
        task=normalized,
        evaluator=evaluator,
    )
    return trainer.fit(train_dataloader, val_dataloader, epochs=epochs)
