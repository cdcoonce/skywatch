"""The default injector set, in the order docs/design.md prescribes."""

from __future__ import annotations

from collections.abc import Sequence

from skywatch.types import Planet
from skywatch.world.glitches import (
    CloudInjector,
    CosmicRayInjector,
    DropoutInjector,
    HotPixelInjector,
    TrackingDriftInjector,
)
from skywatch.world.scene import Injector
from skywatch.world.science import (
    EclipseInjector,
    FlareInjector,
    PulsationInjector,
    TransitInjector,
)


def default_injectors(planets: Sequence[Planet]) -> tuple[list[Injector], list[Injector]]:
    """Return (science, instrument) injectors with each class's default rates."""
    science: list[Injector] = [
        TransitInjector(planets),
        EclipseInjector(),
        PulsationInjector(),
        FlareInjector(),
    ]
    instrument: list[Injector] = [
        CosmicRayInjector(),
        CloudInjector(),
        TrackingDriftInjector(),
        HotPixelInjector(),
        DropoutInjector(),
    ]
    return science, instrument
