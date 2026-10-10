"""Detector interface and registry. Detectors never see truth."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Mapping
from typing import Protocol

from skywatch.types import Detection, LightCurve


class Detector(Protocol):
    name: str

    def detect(self, curves: Mapping[int, LightCurve], cadence_s: int) -> list[Detection]: ...


REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator: add a detector class to REGISTRY under its `name`."""
    REGISTRY[cls.name] = cls
    return cls


def load_all() -> dict[str, type]:
    """Import every module in this package so each @register runs; return REGISTRY."""
    for mod in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{mod.name}")
    return REGISTRY
