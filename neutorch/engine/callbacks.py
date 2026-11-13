"""Training lifecycle hooks for the NeuTorch engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from neutorch.engine.trainer import Trainer


class Callback:
    """Base class for Trainer lifecycle hooks.

    Subclass and override the methods you need. All hooks are no-ops by
    default so unused events incur zero cost. Callbacks receive the
    owning :class:`~neutorch.engine.trainer.Trainer` plus event-specific kwargs.

    Typical use cases: early stopping, checkpointing, learning-rate
    scheduling side-effects, custom metric logging, progress bars.
    """

    def on_train_start(self, trainer: Trainer, **kwargs: Any) -> None:
        """Called once before the first training epoch.

        Args:
            trainer: The owning Trainer instance.
            **kwargs: Extra context supplied by the Trainer.
        """
        pass

    def on_train_end(self, trainer: Trainer, **kwargs: Any) -> None:
        """Called once after the final training epoch.

        Args:
            trainer: The owning Trainer instance.
            **kwargs: Extra context supplied by the Trainer.
        """
        pass

    def on_epoch_start(
        self,
        trainer: Trainer,
        *,
        epoch: int,
        **kwargs: Any,
    ) -> None:
        """Called at the beginning of every epoch.

        Args:
            trainer: The owning Trainer instance.
            epoch: Zero-based epoch index.
            **kwargs: Extra context supplied by the Trainer.
        """
        pass

    def on_epoch_end(
        self,
        trainer: Trainer,
        *,
        epoch: int,
        metrics: Optional[Mapping[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Called at the end of every epoch (after optional validation).

        Args:
            trainer: The owning Trainer instance.
            epoch: Zero-based epoch index that just finished.
            metrics: Aggregated train/val metrics for the epoch.
            **kwargs: Extra context supplied by the Trainer.
        """
        pass

    def on_batch_start(
        self,
        trainer: Trainer,
        *,
        batch: Any,
        batch_idx: int,
        **kwargs: Any,
    ) -> None:
        """Called before processing each training batch.

        Args:
            trainer: The owning Trainer instance.
            batch: Raw batch from the DataLoader (pre-device transfer).
            batch_idx: Index of the batch within the current epoch.
            **kwargs: Extra context supplied by the Trainer.
        """
        pass

    def on_batch_end(
        self,
        trainer: Trainer,
        *,
        batch: Any,
        batch_idx: int,
        outputs: Optional[Mapping[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Called after processing each training batch.

        Args:
            trainer: The owning Trainer instance.
            batch: Batch that was just processed.
            batch_idx: Index of the batch within the current epoch.
            outputs: Step-level metrics returned by ``train_step``.
            **kwargs: Extra context supplied by the Trainer.
        """
        pass

    def on_validation_start(self, trainer: Trainer, **kwargs: Any) -> None:
        """Called once before a validation pass begins.

        Args:
            trainer: The owning Trainer instance.
            **kwargs: Extra context supplied by the Trainer.
        """
        pass

    def on_validation_end(
        self,
        trainer: Trainer,
        *,
        metrics: Optional[Mapping[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Called once after a validation pass completes.

        Args:
            trainer: The owning Trainer instance.
            metrics: Aggregated validation metrics.
            **kwargs: Extra context supplied by the Trainer.
        """
        pass
