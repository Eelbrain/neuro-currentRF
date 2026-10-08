"""Forward model and noise covariance with derived whitened quantities."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from eelbrain import NDVar, Sensor, SourceSpace, Space, VolumeSourceSpace
import mne
import numpy as np
from scipy import linalg

from ._initialization import MNEInitializer
from ._pickle import pickle_state
from ._repr import _forward_summary
from ._typing import FloatArray, NoiseArg

#: Relative eigenvalue threshold below which a direction of the noise covariance is
#: treated as absent. Directions removed exactly (ICA components, SSP projections,
#: the SSS complement) leave eigenvalues at or below ~1e-15 of the largest, while
#: measured noise directions stay above ~1e-8 even for unscaled magnetometer plus
#: gradiometer data, so 1e-12 sits well clear of both; it also caps the whitening
#: gain at 1e6, above MNE-Python's instability warning (condition number 1e10).
_RANK_TOL = 1e-12


def _assert_sensors_equal(
        names: Sequence[str],
        reference: Sequence[str],
        desc: str,
        reference_desc: str,
) -> None:
    """Check that two channel lists are identical, including their order.

    Sensor data is combined by channel position throughout, so anything but an
    exact match silently attributes the data of one channel to another.

    Parameters
    ----------
    names, reference
        Channel lists to compare.
    desc, reference_desc
        What the two lists describe, for the error message.

    Raises
    ------
    ValueError
        If the two lists differ in channels or in their order.
    """
    names, reference = list(names), list(reference)
    if names == reference:
        return
    only_names = [name for name in names if name not in set(reference)]
    only_reference = [name for name in reference if name not in set(names)]
    if only_names or only_reference:
        difference = f"only in {desc}: {only_names or 'none'}; only in {reference_desc}: {only_reference or 'none'}"
    else:
        difference = f"same channels in a different order; {desc} starts with {names[:3]}, {reference_desc} with {reference[:3]}"
    raise ValueError(f"{desc} sensors do not match the {reference_desc} ({difference})")


def _noise_covariance(noise: NoiseArg) -> tuple[FloatArray, list[str]]:
    """Sensor-space noise covariance and its channel names.

    Parameters
    ----------
    noise
        Noise as :class:`mne.Covariance`, or as :class:`eelbrain.NDVar` data
        (typically an empty-room recording) from which the covariance is
        estimated. Bare arrays are not accepted: without channel names, the
        covariance cannot be safely aligned with the lead field.

    Raises
    ------
    TypeError
        If ``noise`` is none of the supported types.
    """
    if isinstance(noise, mne.Covariance):
        data = np.diag(noise.data) if noise['diag'] else noise.data
        return data, list(noise.ch_names)
    elif isinstance(noise, NDVar):
        er = noise.get_data(('sensor', 'time'))
        return np.dot(er, er.T) / er.shape[1], list(noise.get_dim('sensor').names)
    else:
        raise TypeError(f"Invalid noise type: {type(noise)}. Must be NDVar or mne.Covariance.")


@dataclass(eq=False, repr=False)
class ForwardModel:
    """Forward model and noise covariance with derived whitened quantities.

    The lead field and noise covariance are stored as supplied; the whitened
    quantities used by the solver are derived (and recomputed on unpickling)
    rather than stored.  A single instance is shared read-only across
    cross-validation folds.

    Whitening projects sensor space onto the :attr:`rank` leading eigenvectors of
    the noise covariance, so the whitened quantities have ``rank`` rather than
    ``n_sensors`` channels. For a full-rank covariance the two coincide; for a
    rank-deficient one (e.g. after ICA component removal, Maxwell filtering, or
    SSP projection) the directions without noise are dropped, as data cleaned
    the same way carries no signal there either. The rank is the number of
    eigenvalues above ``1e-12`` times the largest, a threshold well clear of both
    the numerical floor of removed directions (~1e-15) and the quietest measured
    noise direction (~1e-8 even for unscaled magnetometer plus gradiometer data).

    Parameters
    ----------
    lead_field
        Forward solution as a 2-D array, shape ``(n_sensors, n_sources)`` or
        ``(n_sensors, n_sources * len(space))`` for free orientation.
    noise_covariance
        Sensor-space noise covariance, shape ``(n_sensors, n_sensors)``, with the
        channels in the order of ``sensor``.
    source
        Source dimension of the forward model.
    sensor
        Sensor dimension of the forward model.
    space
        Orientation dimension (``None`` for fixed orientation).
    """

    lead_field: FloatArray
    noise_covariance: FloatArray
    source: SourceSpace | VolumeSourceSpace
    sensor: Sensor
    space: Space | None
    #: Rank of :attr:`~ncrf.ForwardModel.noise_covariance`, i.e. the number of whitened channels.
    rank: int = field(init=False)
    #: Whitening filter, shape ``(rank, n_sensors)``: the inverse square root of
    #: :attr:`~ncrf.ForwardModel.noise_covariance` restricted to its ``rank`` leading eigenvectors.
    whitening_filter: FloatArray = field(init=False)
    #: Whitened and spectrally normalized :attr:`~ncrf.ForwardModel.lead_field` used by solvers, shape ``(rank, n_sources)``.
    whitened_lead_field: FloatArray = field(init=False)
    #: :attr:`~ncrf.ForwardModel.noise_covariance` transformed by :attr:`~ncrf.ForwardModel.whitening_filter`: the ``(rank, rank)`` identity.
    whitened_noise_covariance: FloatArray = field(init=False)
    #: Spectral norm removed from the whitened lead field.
    lead_field_scaling: float = field(init=False)

    def __post_init__(self) -> None:
        # The lead field and the whitening filter are indexed by position, so a
        # mismatch in either would silently mix up channels or sources.
        n_sensors, n_sources = len(self.sensor), len(self.source) * self.dc
        if self.lead_field.shape != (n_sensors, n_sources):
            raise ValueError(f"lead_field of shape {self.lead_field.shape}; should be {(n_sensors, n_sources)} for {n_sensors} sensors and {len(self.source)} sources with {self.dc} orientation(s)")
        if self.noise_covariance.shape != (n_sensors, n_sensors):
            raise ValueError(f"noise covariance of shape {self.noise_covariance.shape}; should be {(n_sensors, n_sensors)} to match the {n_sensors} sensors of the lead field")
        self._prewhiten()

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {_forward_summary(self)}>'

    @property
    def dc(self) -> int:
        """Number of orientation components per source."""
        return len(self.space) if self.space is not None else 1

    @cached_property
    def mne_initializer(self) -> MNEInitializer:
        """MNE-style initializer for :attr:`~ncrf.ForwardModel.whitened_lead_field`."""
        return MNEInitializer(self.whitened_lead_field)

    def source_block(self, i: int) -> slice:
        """Column/row slice of source ``i``'s orientation components in stacked arrays."""
        dc = self.dc
        return slice(i * dc, (i + 1) * dc)

    def _prewhiten(self) -> None:
        """Compute whitened derived quantities from ``lead_field`` and ``noise_covariance``.

        Writes ``rank``, ``whitening_filter``, ``whitened_lead_field``,
        ``lead_field_scaling``, and ``whitened_noise_covariance``.  Neither
        ``lead_field`` nor ``noise_covariance`` is modified.
        """
        # A channel without noise would be projected out of the whitened space, silently ignoring its data
        flat = np.flatnonzero(np.diag(self.noise_covariance) == 0)
        if flat.size:
            raise ValueError(f"noise covariance has flat channels: {', '.join(self.sensor.names[i] for i in flat)}; exclude them from the data (mark them as bad) or supply noise for them")
        e, v = linalg.eigh(self.noise_covariance)  # ascending eigenvalues
        if e[-1] <= 0:
            raise ValueError("noise covariance has no positive eigenvalues; whitening requires noise in at least one direction")
        rank = self.rank = int((e > _RANK_TOL * e[-1]).sum())
        e, v = e[-rank:], v[:, -rank:]
        self.whitening_filter = v.T / np.sqrt(e)[:, None]
        self.whitened_lead_field = np.dot(self.whitening_filter, self.lead_field)
        # wf @ C @ wf.T is the identity by construction (the kept eigenvectors are orthonormal)
        self.whitened_noise_covariance = np.eye(rank)
        self.lead_field_scaling = linalg.norm(self.whitened_lead_field, 2)
        self.whitened_lead_field /= self.lead_field_scaling

    def __getstate__(self) -> dict[str, Any]:
        # Derived (whitened) quantities are recomputed by _prewhiten() on unpickling.
        return pickle_state(self)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._prewhiten()
