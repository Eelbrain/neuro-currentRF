"""Tests for the sparsity penalties and their proximal operators."""

import numpy as np

from ncrf._fastac import Fasta
from ncrf._penalties import g, g_group, proxg_group, shrink


def test_proximal_operators():
    """The proximal operators leave their input unchanged, and FASTA's ``sg`` is the subgradient of the penalty at the new estimate."""
    # FASTA reads the input again after applying the operator (see sg below)
    for order in ('C', 'F'):
        z = np.asarray(np.random.RandomState(0).normal(size=(6, 4)), order=order)
        before = z.copy()
        out = proxg_group(z, 0.5)
        np.testing.assert_array_equal(z, before)
        assert not np.shares_memory(out, z)
        # shrinkage towards zero, per source (3 orientation components) and time point
        norms = np.linalg.norm(z.reshape(-1, 3, 4), axis=1)
        expected = np.maximum(1 - 0.5 / norms, 0)
        np.testing.assert_allclose(out.reshape(-1, 3, 4), z.reshape(-1, 3, 4) * expected[:, None, :])

    # sg is the subgradient of the penalty at the new estimate, not zero
    rng = np.random.RandomState(0)
    x = rng.normal(size=(6, 4))
    grad = rng.normal(size=(6, 4))
    mu, tau = 0.1, 0.5
    for penalty, prox in [
        (lambda v: g(v, mu), lambda v, t: shrink(v, mu * t)),
        (lambda v: g_group(v, mu), lambda v, t: proxg_group(v, mu * t)),
    ]:
        fasta = Fasta(lambda v: 0.5 * (v ** 2).sum(), penalty, None, prox, beta=0.5)
        z, _, sg, _, _ = fasta._update_coefs(x, tau, grad, np.inf)
        # sg = (x_hat - z) / tau, i.e. the term FASTA balances against the gradient
        np.testing.assert_allclose(sg, (x - tau * grad - z) / tau)
        assert np.abs(sg).max() > 0
