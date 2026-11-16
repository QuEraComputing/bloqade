"""Build architecture specifications.

The builders here produce a :class:`bloqade.lanes.arch.spec.ArchSpec`, the type
the lanes compiler consumes. ``gemini_full`` and ``generic_full`` are
three-zone example architectures built with them.

Two styles are available:

* declarative: describe zones with :class:`ZoneSpec` and a :class:`DeviceLayout`
  in an :class:`ArchBlueprint`, then call :func:`build_arch`;
* imperative: add words and buses one by one with :class:`ZoneBuilder` and
  :class:`ArchBuilder`. ``bloqade.gemini.arch.build.v2`` holds a newer
  spec-shaped ``ArchBuilder``.
"""

from bloqade.gemini.arch.build.topology import (
    SiteTopology as SiteTopology,
    WordTopology as WordTopology,
    MatchingTopology as MatchingTopology,
    InterZoneTopology as InterZoneTopology,
    AllToAllSiteTopology as AllToAllSiteTopology,
    DiagonalWordTopology as DiagonalWordTopology,
    HypercubeSiteTopology as HypercubeSiteTopology,
    HypercubeWordTopology as HypercubeWordTopology,
    TransversalSiteTopology as TransversalSiteTopology,
)
from bloqade.gemini.arch.build.blueprint import (
    ZoneSpec as ZoneSpec,
    ArchResult as ArchResult,
    DeviceLayout as DeviceLayout,
    ArchBlueprint as ArchBlueprint,
    build_arch as build_arch,
)
from bloqade.gemini.arch.build.imperative import (
    ArchBuilder as ArchBuilder,
    ZoneBuilder as ZoneBuilder,
)
from bloqade.gemini.arch.build.word_factory import (
    WordGrid as WordGrid,
    create_zone_words as create_zone_words,
)
