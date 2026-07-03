"""Containers and file I/O for measured RFI-cancellation records.

Experimental acquisitions are just windows of ADC samples: a primary receive
channel and one or more continuously-running reference channels on a shared shot
clock. The gating that the cancellers need -- which samples are transmit,
ringdown, signal gap, or baseline -- is *not* recorded per sample; it is
reconstructed from the pulse-sequence timing metadata (see
``slse_mask_from_metadata`` / ``sorc_mask_from_metadata`` in
:mod:`spin_dynamics.nqr`).

:class:`RFIRecording` bundles the sample windows with a reconstructed
:class:`~spin_dynamics.interference.masks.AcquisitionMask` and free-form metadata,
so a measured record drops straight into the ``(primary, references, mask)``
canceller contract. :func:`save_rfi_recording` / :func:`load_rfi_recording`
round-trip a recording through a NumPy ``.npz`` archive (the mask is stored as its
label array, so no reconstruction is needed on reload).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spin_dynamics.interference.masks import AcquisitionMask


@dataclass(frozen=True)
class RFIRecording:
    """A measured RFI record: ADC sample windows plus a reconstructed mask."""

    primary: np.ndarray
    references: np.ndarray
    sample_rate_hz: float
    mask: AcquisitionMask
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        primary = np.asarray(self.primary)
        primary = primary.reshape(-1)
        if primary.size == 0:
            raise ValueError("primary must contain at least one sample")
        if not np.all(np.isfinite(primary)):
            raise ValueError("primary samples must be finite")

        references = np.asarray(self.references)
        if references.ndim == 1:
            references = references.reshape(1, -1)
        if references.ndim != 2:
            raise ValueError("references must have shape (K, N)")
        if references.shape[0] > 0:
            if references.shape[1] != primary.size:
                raise ValueError("references and primary must share the sample count")
            if not np.all(np.isfinite(references)):
                raise ValueError("reference samples must be finite")

        if not isinstance(self.mask, AcquisitionMask):
            raise TypeError("mask must be an AcquisitionMask")
        if self.mask.num_samples != primary.size:
            raise ValueError("mask length must match the primary sample count")

        sample_rate_hz = float(self.sample_rate_hz)
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive and finite")
        if not np.isclose(sample_rate_hz, self.mask.sample_rate_hz):
            raise ValueError("sample_rate_hz must match the mask sample rate")

        object.__setattr__(self, "primary", primary)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_samples(self) -> int:
        """Number of samples in each channel window."""

        return int(self.primary.size)

    @property
    def num_references(self) -> int:
        """Number of reference channels (may be zero for tracker-only records)."""

        return int(self.references.shape[0])


def save_rfi_recording(path: str | Path, recording: RFIRecording) -> None:
    """Write a recording to a NumPy ``.npz`` archive.

    The mask is stored as its integer label array, so :func:`load_rfi_recording`
    restores the recording exactly without re-deriving the gating.
    """

    path = Path(path)
    np.savez(
        path,
        primary=recording.primary,
        references=recording.references,
        sample_rate_hz=np.float64(recording.sample_rate_hz),
        mask_labels=recording.mask.labels,
        metadata=np.asarray(json.dumps(recording.metadata)),
    )


def load_rfi_recording(path: str | Path) -> RFIRecording:
    """Load a recording previously written by :func:`save_rfi_recording`."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        sample_rate_hz = float(data["sample_rate_hz"])
        mask = AcquisitionMask(sample_rate_hz=sample_rate_hz, labels=data["mask_labels"])
        metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
        return RFIRecording(
            primary=data["primary"],
            references=data["references"],
            sample_rate_hz=sample_rate_hz,
            mask=mask,
            metadata=metadata,
        )
