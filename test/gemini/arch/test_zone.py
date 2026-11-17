"""Tests for zone definition data model."""

import pytest

from bloqade.gemini.arch.build.blueprint import ZoneSpec, DeviceLayout, ArchBlueprint


class TestZoneSpec:
    def test_construction(self) -> None:
        spec = ZoneSpec(num_rows=2, num_cols=4, entangling=True, measurement=True)
        assert spec.num_rows == 2
        assert spec.num_cols == 4
        assert spec.num_words == 8
        assert spec.entangling is True
        assert spec.measurement is True

    def test_defaults(self) -> None:
        spec = ZoneSpec(num_rows=1, num_cols=2)
        assert spec.entangling is False
        assert spec.measurement is True

    def test_num_words_property(self) -> None:
        spec = ZoneSpec(num_rows=3, num_cols=4)
        assert spec.num_words == 12

    def test_frozen(self) -> None:
        spec = ZoneSpec(num_rows=1, num_cols=2)
        with pytest.raises(AttributeError):
            spec.num_rows = 8  # type: ignore[misc]

    def test_equality(self) -> None:
        a = ZoneSpec(num_rows=2, num_cols=4, entangling=True)
        b = ZoneSpec(num_rows=2, num_cols=4, entangling=True)
        c = ZoneSpec(num_rows=2, num_cols=4, entangling=False)
        assert a == b
        assert a != c

    def test_num_rows_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_rows must be >= 1"):
            ZoneSpec(num_rows=0, num_cols=2)

    def test_num_cols_one_raises(self) -> None:
        with pytest.raises(ValueError, match="num_cols must be >= 2"):
            ZoneSpec(num_rows=1, num_cols=1)

    def test_num_cols_odd_raises(self) -> None:
        with pytest.raises(ValueError, match="num_cols must be even"):
            ZoneSpec(num_rows=1, num_cols=3)


class TestDeviceLayout:
    def test_defaults(self) -> None:
        layout = DeviceLayout()
        assert layout.sites_per_word == 5
        assert layout.site_spacing == 10.0
        assert layout.pair_spacing == 10.0
        assert layout.row_spacing == 20.0
        assert layout.zone_gap == 20.0

    def test_custom_values(self) -> None:
        layout = DeviceLayout(sites_per_word=3, site_spacing=5.0, zone_gap=30.0)
        assert layout.sites_per_word == 3
        assert layout.site_spacing == 5.0
        assert layout.zone_gap == 30.0

    def test_frozen(self) -> None:
        layout = DeviceLayout()
        with pytest.raises(AttributeError):
            layout.sites_per_word = 10  # type: ignore[misc]


class TestArchBlueprint:
    def test_valid_single_zone(self) -> None:
        bp = ArchBlueprint(
            zones={"proc": ZoneSpec(num_rows=2, num_cols=4, entangling=True)}
        )
        assert bp.words_per_zone == 8
        assert bp.zone_names == ("proc",)

    def test_valid_two_zones(self) -> None:
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(num_rows=2, num_cols=4, entangling=True),
                "mem": ZoneSpec(num_rows=2, num_cols=4),
            }
        )
        # Shared template: all zones reuse ONE 8-word list.
        assert bp.words_per_zone == 8
        assert bp.zone_names == ("proc", "mem")

    def test_valid_three_zones(self) -> None:
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(num_rows=2, num_cols=4, entangling=True),
                "buffer": ZoneSpec(num_rows=2, num_cols=4),
                "mem": ZoneSpec(num_rows=2, num_cols=4),
            }
        )
        # Shared template: 3 zones reuse ONE 8-word list.
        assert bp.words_per_zone == 8
        assert bp.zone_names == ("proc", "buffer", "mem")

    def test_empty_zones_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one zone"):
            ArchBlueprint(zones={})

    def test_unequal_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match="same grid dimensions"):
            ArchBlueprint(
                zones={
                    "proc": ZoneSpec(num_rows=2, num_cols=4),
                    "mem": ZoneSpec(num_rows=4, num_cols=2),
                }
            )

    def test_zone_ordering_preserved(self) -> None:
        bp = ArchBlueprint(
            zones={
                "mem": ZoneSpec(num_rows=1, num_cols=2),
                "proc": ZoneSpec(num_rows=1, num_cols=2),
                "buffer": ZoneSpec(num_rows=1, num_cols=2),
            }
        )
        assert bp.zone_names == ("mem", "proc", "buffer")

    def test_default_layout(self) -> None:
        bp = ArchBlueprint(zones={"proc": ZoneSpec(num_rows=1, num_cols=2)})
        assert bp.layout == DeviceLayout()

    def test_custom_layout(self) -> None:
        layout = DeviceLayout(sites_per_word=3, zone_gap=50.0)
        bp = ArchBlueprint(
            zones={"proc": ZoneSpec(num_rows=1, num_cols=2)},
            layout=layout,
        )
        assert bp.layout.sites_per_word == 3
        assert bp.layout.zone_gap == 50.0

    def test_frozen(self) -> None:
        bp = ArchBlueprint(zones={"proc": ZoneSpec(num_rows=1, num_cols=2)})
        with pytest.raises(AttributeError):
            bp.zones = {}  # type: ignore[misc]
