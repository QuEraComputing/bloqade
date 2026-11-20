"""Architecture construction tooling.

Public API for building ``ArchSpec`` instances. Two layers:

- :mod:`.imperative` — low-level ``ZoneBuilder`` / ``ArchBuilder`` for
  hand-crafted zones.
- :mod:`.blueprint` — high-level ``build_arch(blueprint)`` entry point
  that consumes a declarative ``ArchBlueprint`` (zones + ``DeviceLayout``)
  and emits a validated ``ArchSpec``.

Plus :mod:`.topology` (connectivity protocol implementations) and
:mod:`.word_factory` (``WordGrid`` helper used during zone assembly).
"""

from bloqade.gemini.arch.build.blueprint import (
    ArchBlueprint as ArchBlueprint,
    ArchResult as ArchResult,
    DeviceLayout as DeviceLayout,
    ZoneSpec as ZoneSpec,
    build_arch as build_arch,
)
from bloqade.gemini.arch.build.imperative import (
    ArchBuilder as ArchBuilder,
    ZoneBuilder as ZoneBuilder,
)
from bloqade.gemini.arch.build.topology import (
    AllToAllSiteTopology as AllToAllSiteTopology,
    DiagonalWordTopology as DiagonalWordTopology,
    HypercubeSiteTopology as HypercubeSiteTopology,
    HypercubeWordTopology as HypercubeWordTopology,
    InterZoneTopology as InterZoneTopology,
    MatchingTopology as MatchingTopology,
    SiteTopology as SiteTopology,
    TransversalSiteTopology as TransversalSiteTopology,
    WordTopology as WordTopology,
)
from bloqade.gemini.arch.build.word_factory import (
    WordGrid as WordGrid,
    create_zone_words as create_zone_words,
)
