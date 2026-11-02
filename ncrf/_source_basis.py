"""Gaussian basis over the source space, the spatial counterpart of the TRF basis.

With a :class:`~ncrf.SourceBasis`, the TRF coefficients are fit over spatial atoms
rather than over the individual sources: ``theta = Phi @ alpha``, where each atom
is a Gaussian bump over the source space carrying all orientation components of
its sources (``Phi = phi ⊗ I_dc`` for the scalar basis ``phi``). The sparsity
penalty then acts on the atom coefficients, so that the estimated response
functions are sums of a few smooth, local patches. Since ``Phi`` mixes sources but
never orientation components, the orientation groups of the penalty, and its
invariance to rotations, carry over from sources to atoms.

The solver sees the atoms as virtual sources with lead field ``L @ Phi``
(:attr:`~ncrf.ForwardModel.whitened_lead_field`); the fitted coefficients are
expanded back onto the sources with :meth:`~ncrf.ForwardModel.expand`.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from math import log, sqrt

from eelbrain import SourceSpace, VolumeSourceSpace
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from ._typing import FloatArray, IndexArray

#: Support of the Gaussian kernel, in units of its standard deviation: atoms are
#: truncated where the kernel drops below 1e-6 of its peak (about 5.3 std), which
#: keeps the basis sparse without measurably changing it.
_SUPPORT = sqrt(2 * log(1e6))

#: Atom spacing, in units of the kernel standard deviation, above which a warning
#: is issued. Two atoms this far apart sum to about 0.9 at their midpoint, so a
#: wider spacing cannot represent a uniform patch of activity without ripples
#: (the TRF basis' default stride of one sample at 100 Hz corresponds to 1.2).
_MAX_SPACING = 2.5


@dataclass(frozen=True, eq=False, repr=False)
class SourceBasis:
    """Gaussian basis over the sources of a source space.

    Use :meth:`from_source` to construct the basis for a source space.

    Parameters
    ----------
    basis
        Scalar basis, shape ``(n_sources, n_atoms)``: column ``j`` is the Gaussian
        atom centered on source ``centers[j]``, with peak value 1.
    std
        Standard deviation of the Gaussian atoms in meters.
    stride
        Spacing between neighboring atom centers in grid steps, as passed to
        :meth:`from_source`.
    centers
        Index of the source each atom is centered on.
    """

    basis: sparse.csr_array
    std: float
    stride: int
    centers: IndexArray

    @classmethod
    def from_source(
            cls,
            source: SourceSpace | VolumeSourceSpace,
            std: float,
            stride: int = 1,
    ) -> SourceBasis:
        """Construct the basis for a source space.

        The source positions are read from the MNE source space file of
        ``source`` (see :meth:`eelbrain.SourceSpace.get_source_space`).

        Parameters
        ----------
        source
            Source space of the lead field.

            - :class:`~eelbrain.VolumeSourceSpace`: atoms are Gaussian in the
              Euclidean distance between source positions.
            - :class:`~eelbrain.SourceSpace` (cortical surface): atoms are
              Gaussian in the geodesic distance along the cortical surface, which
              the source space file must contain (see
              :func:`mne.add_source_space_distances`) up to at least ``5.3 * std``.

            An atom never extends across hemispheres.
        std
            Standard deviation of the Gaussian atoms in meters; at most the
            extent of the source space.
        stride
            Spacing between neighboring atom centers, in grid steps: with the
            default of ``1``, every source is an atom center, and larger values
            center atoms on every ``stride``-th grid point along each axis. Only
            ``1`` is supported for cortical surfaces.

        Raises
        ------
        ValueError
            If ``std`` or ``stride`` is invalid, ``std`` exceeds the extent of the
            source space (typically a value in millimeters rather than meters),
            or a surface source space lacks geodesic distances.
        NotImplementedError
            For ``stride > 1`` with a cortical surface source space.
        """
        if not std > 0 or not np.isfinite(std):
            raise ValueError(f"{std=}: need a positive, finite standard deviation (in meters)")
        if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or stride < 1:
            raise ValueError(f"{stride=}: need an integer >= 1")
        if not isinstance(source, (SourceSpace, VolumeSourceSpace)):
            raise TypeError(f"{source=}: need a SourceSpace or VolumeSourceSpace")
        # Atoms wider than the source space are nearly constant over it, and their
        # support would cover all source pairs, so check before building them
        extent = np.ptp(source.coordinates, axis=0).max()
        if std > extent:
            raise ValueError(f"{std=}: larger than the source space, which spans {extent * 1000:.0f} mm; std is in meters")
        radius = _SUPPORT * std
        if isinstance(source, VolumeSourceSpace):
            centers, distances = _volume_distances(source, stride, radius)
        elif isinstance(source, SourceSpace):
            if stride != 1:
                raise NotImplementedError(f"{stride=} for a cortical surface source space: only stride=1 is supported")
            centers, distances = _surface_distances(source, radius)
        # distances hold an explicit 0 at each atom's center, so each atom peaks at 1 there
        basis = distances.copy()
        basis.data = np.exp(-basis.data ** 2 / (2 * std ** 2))
        _check_spacing(distances, centers, std)
        return cls(basis=basis.tocsr(), std=float(std), stride=int(stride), centers=centers)

    def __repr__(self) -> str:
        n_sources, n_atoms = self.basis.shape
        return f'<{type(self).__name__}: {n_sources=}, {n_atoms=}, std={self.std:g}, stride={self.stride}>'

    @property
    def n_sources(self) -> int:
        """Number of sources the basis is defined over."""
        return self.basis.shape[0]

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return self.basis.shape[1]

    def reduce(self, lead_field: FloatArray, dc: int) -> FloatArray:
        """Lead field of the atoms, ``lead_field @ (basis ⊗ I_dc)``.

        Parameters
        ----------
        lead_field
            Lead field over the sources, shape ``(n_channels, n_sources * dc)``
            with each source's ``dc`` orientation components in adjacent columns.
        dc
            Number of orientation components per source.

        Returns
        -------
        ndarray
            Lead field over the atoms, shape ``(n_channels, n_atoms * dc)``, with
            each atom's orientation components in adjacent columns.
        """
        n_channels = lead_field.shape[0]
        g = lead_field.reshape(n_channels, self.n_sources, dc).transpose(1, 0, 2).reshape(self.n_sources, -1)
        g = self.basis.T @ g  # (n_atoms, n_channels * dc)
        return g.reshape(self.n_atoms, n_channels, dc).transpose(1, 0, 2).reshape(n_channels, -1)

    def expand(self, coefficients: FloatArray, dc: int) -> FloatArray:
        """Expand atom coefficients onto the sources, ``(basis ⊗ I_dc) @ coefficients``.

        Parameters
        ----------
        coefficients
            Coefficients over the atoms, shape ``(n_atoms * dc, n_columns)``, with
            each atom's orientation components in adjacent rows.
        dc
            Number of orientation components per source.

        Returns
        -------
        ndarray
            Coefficients over the sources, shape ``(n_sources * dc, n_columns)``.
        """
        n_columns = coefficients.shape[1]
        x = self.basis @ coefficients.reshape(self.n_atoms, dc * n_columns)
        return x.reshape(self.n_sources * dc, n_columns)

    def equals(self, other: SourceBasis | None) -> bool:
        """Whether ``other`` is the same basis (``None`` is no basis)."""
        if other is self:
            return True
        if other is None or other.basis.shape != self.basis.shape:
            return False
        return (other.basis != self.basis).nnz == 0


def _volume_distances(
        source: VolumeSourceSpace,
        stride: int,
        radius: float,
) -> tuple[IndexArray, sparse.coo_array]:
    """Atom centers and source-to-center distances within ``radius``, for a volume source space.

    The distances are explicit for every pair within ``radius``, including the 0
    of each atom's center. The grid indices of the sources are recovered from their positions and the
    grid spacing (the source space's grade, in mm).
    """
    positions = source.coordinates
    grid = (positions - positions.min(0)) / (source.grade / 1000)
    index = np.round(grid).astype(np.int64)
    if np.abs(grid - index).max() > 1e-3:
        raise ValueError(f"{source=}: source positions do not lie on a regular grid with {source.grade} mm spacing")
    centers = np.flatnonzero((index % stride == 0).all(1))
    if len(centers) == 0:
        raise ValueError(f"{stride=}: too coarse for {source}, which leaves no atom centers")
    pairs = cKDTree(positions[centers]).query_ball_point(positions, radius)
    rows = np.repeat(np.arange(len(positions)), [len(p) for p in pairs])
    cols = np.concatenate(pairs).astype(np.int64)
    # sources on opposite sides of the midline belong to different hemispheres
    keep = positions[rows, 0] * positions[centers[cols], 0] >= 0
    rows, cols = rows[keep], cols[keep]
    distances = np.linalg.norm(positions[rows] - positions[centers[cols]], axis=1)
    return centers, sparse.coo_array((distances, (rows, cols)), shape=(len(positions), len(centers)))


def _surface_distances(
        source: SourceSpace,
        radius: float,
) -> tuple[IndexArray, sparse.coo_array]:
    """Atom centers (all sources) and geodesic source-to-source distances within ``radius``, for a cortical surface.

    The distances are explicit for every pair within ``radius``, including the 0
    of each atom's center, which MNE does not store.
    """
    blocks = []
    for vertices, ss in zip(source.vertices, source.get_source_space()):
        if ss['dist'] is None:
            raise ValueError(f"{source=}: the source space has no geodesic distances; add them with mne.add_source_space_distances(src, dist_limit={radius:.3g}) and save it")
        dist_limit = float(np.squeeze(ss['dist_limit']))
        # dist_limit is stored in single precision
        if dist_limit < radius * (1 - 1e-6):
            raise ValueError(f"{source=}: geodesic distances only up to {dist_limit * 1000:g} mm, the basis needs them up to {radius * 1000:.3g} mm ({_SUPPORT:.2g} std); recompute them with mne.add_source_space_distances(src, dist_limit={radius:.3g})")
        dist = sparse.csr_array(ss['dist'])[vertices][:, vertices].tocoo()
        keep = (dist.data <= radius) & (dist.row != dist.col)
        diagonal = np.arange(len(vertices))
        rows = np.concatenate([dist.row[keep], diagonal])
        cols = np.concatenate([dist.col[keep], diagonal])
        data = np.concatenate([dist.data[keep], np.zeros(len(vertices))])
        blocks.append(sparse.coo_array((data, (rows, cols)), shape=dist.shape))
    # hemispheres are separate blocks: no geodesic path connects them
    distances = sparse.block_diag(blocks, format='coo')
    return np.arange(distances.shape[0]), distances


def _check_spacing(distances: sparse.coo_array, centers: IndexArray, std: float) -> None:
    """Warn if neighboring atoms are too far apart, relative to ``std``, to overlap smoothly.

    The spacing is the median distance from an atom's center to the nearest other
    center, ignoring centers beyond the kernel support (an atom without any
    counts as infinitely far).
    """
    distances = sparse.csr_array(distances)[centers].tocsc()  # center-to-center
    nearest = []
    for j in range(distances.shape[1]):
        d = distances.data[distances.indptr[j]:distances.indptr[j + 1]]
        d = d[d > 0]
        nearest.append(d.min() if len(d) else np.inf)
    spacing = np.median(nearest)
    if spacing > _MAX_SPACING * std:
        logging.getLogger(__name__).warning(f"source basis atoms are {spacing * 1000:.3g} mm apart, {spacing / std:.2g} times their std of {std * 1000:.3g} mm; neighboring atoms barely overlap, so the basis cannot represent smooth activity between them (consider a larger std or smaller stride)")
