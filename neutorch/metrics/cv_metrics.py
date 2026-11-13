"""Computer Vision metric wrappers built on torchmetrics."""

from __future__ import annotations

import logging
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchmetrics.detection import MeanAveragePrecision

logger = logging.getLogger(__name__)


def _resolve_device(
    device: Optional[Union[str, torch.device]] = None,
) -> torch.device:
    """Prefer CUDA when available so metric state stays on GPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_tree(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in nested structures onto ``device``."""
    if isinstance(obj, Tensor):
        return obj if obj.device == device else obj.to(device, non_blocking=True)
    if isinstance(obj, Mapping):
        return {key: _move_tree(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_move_tree(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tree(value, device) for value in obj)
    return obj


def _tensor_to_float(value: Tensor) -> float:
    """Detach a scalar (or mean-reduced) tensor for logging."""
    if value.numel() == 1:
        return float(value.detach().item())
    return float(value.detach().float().mean().item())


class _BaseEvaluator:
    """Shared update / compute / reset / logging helpers for CV metrics."""

    metric: Metric

    def __init__(
        self,
        *,
        device: Optional[Union[str, torch.device]] = None,
        writer: Optional[SummaryWriter] = None,
        log_prefix: str = "val/",
    ) -> None:
        self.device = _resolve_device(device)
        self.writer = writer
        self.log_prefix = log_prefix

    def _place_metric(self, metric: Metric) -> Metric:
        """Move metric module + buffers onto the evaluator device."""
        return metric.to(self.device)

    def update(self, preds: Any, targets: Any) -> None:
        """Accumulate one batch while keeping tensors on the metric device."""
        raise NotImplementedError

    def compute(
        self,
        *,
        log: bool = True,
        step: Optional[int] = None,
        reset: bool = True,
    ) -> Mapping[str, float]:
        """Compute epoch metrics, optionally log, then reset state.

        Args:
            log: When ``True``, emit scalars via the logger / TensorBoard.
            step: Optional TensorBoard global step.
            reset: When ``True`` (default), clear metric state for the
                next epoch after computing.

        Returns:
            Flat mapping of metric name → Python float.
        """
        raw = self.metric.compute()
        results = self._normalize_results(raw)

        if log:
            self._log_results(results, step=step)

        if reset:
            self.metric.reset()

        return results

    def reset(self) -> None:
        """Clear accumulated metric state without computing."""
        self.metric.reset()

    def to(self, device: Union[str, torch.device]) -> _BaseEvaluator:
        """Move metric state to ``device`` and return ``self``."""
        self.device = torch.device(device)
        self.metric = self.metric.to(self.device)
        return self

    def _normalize_results(self, raw: Any) -> dict[str, float]:
        """Convert torchmetrics output into a flat float dict."""
        if isinstance(raw, Mapping):
            out: dict[str, float] = {}
            for key, value in raw.items():
                if isinstance(value, Tensor):
                    if value.numel() == 0:
                        continue
                    # Skip non-scalar diagnostics (e.g. per-class vectors
                    # already summarized elsewhere) unless scalar-like.
                    if value.numel() == 1 or value.ndim == 0:
                        out[str(key)] = _tensor_to_float(value)
                    else:
                        out[str(key)] = _tensor_to_float(value)
                elif isinstance(value, (int, float)):
                    out[str(key)] = float(value)
            return out
        if isinstance(raw, Tensor):
            return {"value": _tensor_to_float(raw)}
        return {"value": float(raw)}

    def _log_results(
        self,
        results: Mapping[str, float],
        *,
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to the module logger and optional TensorBoard writer."""
        summary = ", ".join(f"{name}={value:.4f}" for name, value in results.items())
        logger.info("%s%s", self.log_prefix, summary)

        if self.writer is None:
            return
        log_step = 0 if step is None else step
        for name, value in results.items():
            self.writer.add_scalar(f"{self.log_prefix}{name}", value, log_step)


class DetectionEvaluator(_BaseEvaluator):
    """Object-detection evaluator wrapping ``MeanAveragePrecision``.

    Accumulates COCO-style predictions on the configured device across
    an epoch. Call :meth:`compute` at epoch end to obtain mAP / mAR,
    log them, and reset for the next epoch.

    Expected ``update`` inputs (torchmetrics detection format)::

        preds = [{"boxes": FloatTensor[N,4], "scores": FloatTensor[N],
                  "labels": IntTensor[N]}, ...]
        targets = [{"boxes": FloatTensor[M,4], "labels": IntTensor[M]}, ...]

    Args:
        box_format: Box encoding expected by the metric (``"xyxy"``,
            ``"xywh"``, ``"cxcywh"``).
        iou_type: ``"bbox"`` or ``"segm"``.
        iou_thresholds: Optional IoU thresholds; ``None`` uses the COCO
            default range.
        class_metrics: When ``True``, also compute per-class mAP.
        device: Device for metric state. Defaults to CUDA when available.
        writer: Optional TensorBoard writer used by :meth:`compute`.
        log_prefix: Tag prefix for logged scalars.
        kwargs: Forwarded to :class:`~torchmetrics.detection.MeanAveragePrecision`.
    """

    def __init__(
        self,
        *,
        box_format: str = "xyxy",
        iou_type: str = "bbox",
        iou_thresholds: Optional[Sequence[float]] = None,
        class_metrics: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        writer: Optional[SummaryWriter] = None,
        log_prefix: str = "val/",
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, writer=writer, log_prefix=log_prefix)
        metric = MeanAveragePrecision(
            box_format=box_format,
            iou_type=iou_type,
            iou_thresholds=list(iou_thresholds) if iou_thresholds is not None else None,
            class_metrics=class_metrics,
            **kwargs,
        )
        self.metric = self._place_metric(metric)

    def update(
        self,
        preds: Sequence[Mapping[str, Tensor]],
        targets: Sequence[Mapping[str, Tensor]],
    ) -> None:
        """Accumulate a detection batch without host sync.

        Args:
            preds: Per-image prediction dicts (``boxes``, ``scores``, ``labels``).
            targets: Per-image target dicts (``boxes``, ``labels``).
        """
        preds_dev = _move_tree(list(preds), self.device)
        targets_dev = _move_tree(list(targets), self.device)
        self.metric.update(preds_dev, targets_dev)

    def _normalize_results(self, raw: Any) -> dict[str, float]:
        """Prefer the primary mAP scalars from the COCO-style summary."""
        results = super()._normalize_results(raw)
        # Keep a stable primary key for Trainer / callbacks.
        if "map" in results and "mAP" not in results:
            results["mAP"] = results["map"]
        if "map_50" in results and "mAP_50" not in results:
            results["mAP_50"] = results["map_50"]
        return results


class SegmentationEvaluator(_BaseEvaluator):
    """Semantic-segmentation evaluator wrapping multiclass IoU (Jaccard).

    Keeps the Jaccard state on GPU across batches. Call :meth:`compute`
    at epoch end to obtain mean IoU, log it, and reset.

    ``update`` accepts:
        * ``preds``: ``(N, H, W)`` class indices **or** ``(N, C, H, W)``
          logits / probabilities (argmax applied automatically).
        * ``targets``: ``(N, H, W)`` integer class masks.

    Args:
        num_classes: Number of classes including background where used.
        average: Reduction over classes (``"macro"``, ``"micro"``,
            ``"weighted"``, ``"none"``).
        ignore_index: Optional label value ignored in the IoU computation.
        device: Device for metric state. Defaults to CUDA when available.
        writer: Optional TensorBoard writer used by :meth:`compute`.
        log_prefix: Tag prefix for logged scalars.
        kwargs: Forwarded to
            :class:`~torchmetrics.classification.MulticlassJaccardIndex`.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        average: str = "macro",
        ignore_index: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        writer: Optional[SummaryWriter] = None,
        log_prefix: str = "val/",
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, writer=writer, log_prefix=log_prefix)
        self.num_classes = num_classes
        metric_kwargs: MutableMapping[str, Any] = {
            "num_classes": num_classes,
            "average": average,
            **kwargs,
        }
        if ignore_index is not None:
            metric_kwargs["ignore_index"] = ignore_index
        metric = MulticlassJaccardIndex(**metric_kwargs)
        self.metric = self._place_metric(metric)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Accumulate a segmentation batch without host sync.

        Args:
            preds: Logits ``(N, C, H, W)`` or class maps ``(N, H, W)``.
            targets: Ground-truth class maps ``(N, H, W)``.
        """
        preds = preds if preds.device == self.device else preds.to(self.device, non_blocking=True)
        targets = (
            targets
            if targets.device == self.device
            else targets.to(self.device, non_blocking=True)
        )

        if preds.ndim == targets.ndim + 1:
            preds = preds.argmax(dim=1)

        self.metric.update(preds, targets)

    def _normalize_results(self, raw: Any) -> dict[str, float]:
        """Expose a stable ``iou`` / ``mIoU`` key for callers."""
        if isinstance(raw, Tensor):
            value = _tensor_to_float(raw)
            return {"iou": value, "mIoU": value}
        results = super()._normalize_results(raw)
        if "iou" not in results and results:
            # MulticlassJaccardIndex returns a single tensor; Mapping path
            # is uncommon but keep a primary alias if present.
            first = next(iter(results.values()))
            results["iou"] = first
        if "iou" in results:
            results.setdefault("mIoU", results["iou"])
        return results


class ClassificationEvaluator(_BaseEvaluator):
    """Image-classification evaluator wrapping multiclass accuracy.

    ``update`` accepts logits ``(N, C)`` (argmax applied automatically) or
    already-argmaxed class indices ``(N,)``, with integer targets ``(N,)``.

    Args:
        num_classes: Number of classes.
        average: Accuracy reduction (``"micro"``, ``"macro"``, ``"weighted"``).
        top_k: Top-k accuracy parameter.
        device: Device for metric state. Defaults to CUDA when available.
        writer: Optional TensorBoard writer used by :meth:`compute`.
        log_prefix: Tag prefix for logged scalars.
        kwargs: Forwarded to
            :class:`~torchmetrics.classification.MulticlassAccuracy`.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        average: str = "micro",
        top_k: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        writer: Optional[SummaryWriter] = None,
        log_prefix: str = "val/",
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, writer=writer, log_prefix=log_prefix)
        self.num_classes = num_classes
        metric = MulticlassAccuracy(
            num_classes=num_classes,
            average=average,
            top_k=top_k,
            **kwargs,
        )
        self.metric = self._place_metric(metric)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Accumulate a classification batch without host sync."""
        preds = preds if preds.device == self.device else preds.to(self.device, non_blocking=True)
        targets = (
            targets
            if targets.device == self.device
            else targets.to(self.device, non_blocking=True)
        )
        self.metric.update(preds, targets)

    def _normalize_results(self, raw: Any) -> dict[str, float]:
        if isinstance(raw, Tensor):
            value = _tensor_to_float(raw)
            return {"accuracy": value, "acc": value}
        results = super()._normalize_results(raw)
        if "accuracy" not in results and results:
            first = next(iter(results.values()))
            results["accuracy"] = first
        if "accuracy" in results:
            results.setdefault("acc", results["accuracy"])
        return results


def build_evaluator(
    task: str,
    *,
    num_classes: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    writer: Optional[SummaryWriter] = None,
    **kwargs: Any,
) -> _BaseEvaluator:
    """Construct the evaluator matching a vision ``task`` string.

    Args:
        task: ``"detection"``, ``"segmentation"``, or ``"classification"``.
        num_classes: Required for segmentation and classification.
        device: Device for metric state.
        writer: Optional TensorBoard writer attached to the evaluator.
        **kwargs: Forwarded to the concrete evaluator constructor.

    Returns:
        A ready-to-use evaluator instance.
    """
    normalized = task.strip().lower()
    common = {"device": device, "writer": writer, **kwargs}

    if normalized == "detection":
        return DetectionEvaluator(**common)
    if normalized == "segmentation":
        if num_classes is None:
            raise ValueError("num_classes is required for task='segmentation'.")
        return SegmentationEvaluator(num_classes, **common)
    if normalized == "classification":
        if num_classes is None:
            raise ValueError("num_classes is required for task='classification'.")
        return ClassificationEvaluator(num_classes, **common)
    raise ValueError(
        f"Unknown task {task!r}. Expected 'detection', 'segmentation', "
        "or 'classification'."
    )


class CVMetrics:
    """Backward-compatible facade bundling detection + segmentation metrics.

    Prefer :class:`DetectionEvaluator` or :class:`SegmentationEvaluator`
    directly for new code.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        task: str = "detection",
        iou_thresholds: Union[float, Sequence[float]] = 0.5,
        average: str = "macro",
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.device = _resolve_device(device)
        thresholds: Optional[Sequence[float]]
        if isinstance(iou_thresholds, (int, float)):
            thresholds = [float(iou_thresholds)]
        else:
            thresholds = list(iou_thresholds)

        self.detection: Optional[DetectionEvaluator] = None
        self.segmentation: Optional[SegmentationEvaluator] = None

        if task in ("detection", "both"):
            self.detection = DetectionEvaluator(
                iou_thresholds=thresholds,
                device=self.device,
                **kwargs,
            )
        if task in ("segmentation", "multiclass", "both"):
            self.segmentation = SegmentationEvaluator(
                num_classes=num_classes,
                average=average,
                device=self.device,
            )

        self.map_metric: Optional[Metric] = (
            self.detection.metric if self.detection is not None else None
        )
        self.iou_metric: Optional[Metric] = (
            self.segmentation.metric if self.segmentation is not None else None
        )

    def setup(self) -> None:
        """No-op; metrics are constructed in ``__init__``."""
        return None

    def update(self, preds: Any, targets: Any) -> None:
        """Delegate to the active evaluator(s)."""
        if self.detection is not None and self.task in ("detection", "both"):
            self.detection.update(preds, targets)
        if self.segmentation is not None and self.task in (
            "segmentation",
            "multiclass",
            "both",
        ):
            self.segmentation.update(preds, targets)

    def compute(self) -> Mapping[str, float]:
        """Compute, log, and reset all active evaluators."""
        results: dict[str, float] = {}
        if self.detection is not None:
            results.update(self.detection.compute())
        if self.segmentation is not None:
            results.update(self.segmentation.compute())
        return results

    def reset(self) -> None:
        """Reset all active evaluators."""
        if self.detection is not None:
            self.detection.reset()
        if self.segmentation is not None:
            self.segmentation.reset()

    def to(self, device: Union[str, torch.device]) -> CVMetrics:
        """Move active evaluators to ``device``."""
        self.device = torch.device(device)
        if self.detection is not None:
            self.detection.to(self.device)
            self.map_metric = self.detection.metric
        if self.segmentation is not None:
            self.segmentation.to(self.device)
            self.iou_metric = self.segmentation.metric
        return self

    @property
    def map(self) -> Optional[Metric]:
        """Underlying mAP metric, if configured."""
        return self.map_metric

    @property
    def iou(self) -> Optional[Metric]:
        """Underlying IoU metric, if configured."""
        return self.iou_metric
