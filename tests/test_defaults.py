from skywatch.types import Planet
from skywatch.world.defaults import default_injectors


def _planets():
    return [Planet(0, 0, 86400.0, 100.0, 0.01, 7200.0)]


def test_order_of_kinds():
    science, instrument = default_injectors(_planets())
    assert [i.kind for i in science] == ["transit", "eclipse", "pulsation", "flare"]
    assert [i.kind for i in instrument] == [
        "cosmic_ray",
        "cloud",
        "tracking_drift",
        "hot_pixel",
        "dropout",
    ]


def test_kinds_unique_and_protocol():
    science, instrument = default_injectors(_planets())
    kinds = [i.kind for i in [*science, *instrument]]
    assert len(set(kinds)) == len(kinds)
    for inj in [*science, *instrument]:
        assert isinstance(inj.kind, str)
        assert callable(inj.inject)
