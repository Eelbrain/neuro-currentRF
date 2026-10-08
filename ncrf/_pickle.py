"""Shared ``__getstate__`` helper for classes with derived attributes."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from functools import cached_property
from typing import Any


def pickle_state(obj: object) -> dict[str, Any]:
    """State dict without derived attributes, for ``__getstate__``.

    Drops every attribute backed by a ``cached_property`` (recomputed on demand)
    and every ``init=False`` dataclass field (recomputed in ``__setstate__``), so
    that adding a derived attribute can never silently grow the pickle — e.g.
    ship cached Gram matrices to every cross-validation worker.
    """
    cls = type(obj)
    derived = {f.name for f in fields(obj) if not f.init} if is_dataclass(obj) else frozenset()
    return {key: value for key, value in obj.__dict__.items() if key not in derived and not isinstance(getattr(cls, key, None), cached_property)}
