# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: kirin-workspace (3.12.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Extending the site buses of the Gemini physical architecture
#
# Given an architecture, add buses to it — without disturbing what is
# already there.
#
# The Gemini physical spec ships with three site buses, the three
# dimensions of a hypercube over a word's eight sites. That is enough to
# get an atom from any site to any other, but a move between distant sites
# takes several hops. Here we add a fourth bus that does site 0 → site 7
# in one move, and check that nothing else about the architecture changed.
#
# The thing to watch is the transport paths. The bundled spec's lane
# geometry is *calibrated hardware data* — it came from the machine, not
# from the builder's path search, which cannot reproduce it. Extending the
# architecture must leave every one of those paths byte-for-byte intact and
# route only the bus we just added.

# %%
from bloqade.lanes.bytecode.encoding import MoveType, Direction
from bloqade.lanes.arch.gemini.physical import get_arch_spec

from bloqade.gemini.arch import ArchBuilder

spec = get_arch_spec()
zone = spec.zones[0]

print(f"zone {zone.name!r}: {len(spec.words)} words of {spec.sites_per_word} sites")
print(f"words carrying site buses: {list(zone.words_with_site_buses)}")
print("site buses (the hypercube dimensions):")
for bus_id, bus in enumerate(zone.site_buses):
    print(f"  {bus_id}: {list(bus.src)} -> {list(bus.dst)}")
print(f"calibrated transport paths: {len(spec.paths)}")

# %% [markdown]
# ## Restore the spec into a builder
#
# `from_spec` rebuilds the whole architecture — the shared word template,
# the zone's coordinates, its buses and its participation sets. Clearances
# are the one thing an `ArchSpec` does not carry: they are inputs to the
# path search rather than properties of the machine, so we supply them
# here. They are used *only* to route buses we add; everything already in
# the spec keeps the paths it arrived with.

# %%
builder = ArchBuilder.from_spec(spec, x_clearance=0.5, y_clearance=1.0)

# %% [markdown]
# ## Add the long-range site bus
#
# Site 0 and site 7 sit at opposite ends of the word. Reaching one from the
# other over the hypercube takes three moves (0 → 1 → 3 → 7); one direct
# bus does it in a single AOD sweep.
#
# The builder checks the move is something an AOD can actually perform
# before accepting it: every atom's displacement must depend only on its
# row and column, and tones may not cross.

# %%
LONG_RANGE = ([0], [7])
builder.add_site_bus(zone.name, src=LONG_RANGE[0], dst=LONG_RANGE[1])
extended = builder.build(
    feed_forward=spec.feed_forward, atom_reloading=spec.atom_reloading
)

print("site buses after extending:")
for bus_id, bus in enumerate(extended.zones[0].site_buses):
    marker = "  <- new" if bus_id == len(zone.site_buses) else ""
    print(f"  {bus_id}: {list(bus.src)} -> {list(bus.dst)}{marker}")

# %% [markdown]
# ## The calibrated paths are untouched
#
# Every lane that existed before must come back identical. This is the
# property the whole exercise turns on: lane durations are measured off
# these segment lengths, so re-routing them as a side effect of adding a
# bus would quietly change fidelity estimates for the entire machine.

# %%
preserved = sum(1 for lane, path in spec.paths.items() if extended.paths[lane] == path)
print(f"calibrated lanes preserved: {preserved}/{len(spec.paths)}")
assert preserved == len(spec.paths), "extending must not re-route existing lanes"

new_lanes = {lane for lane in extended.paths if lane not in spec.paths}
new_bus_id = len(zone.site_buses)
assert all(
    lane.move_type == MoveType.SITE and lane.bus_id == new_bus_id for lane in new_lanes
), "only the new bus should gain lanes"
print(f"lanes routed for the new bus: {len(new_lanes)}")

# %% [markdown]
# The new lanes cover exactly the words that participate in site-bus
# transport — the odd-numbered words, on this architecture. A word that
# sits site buses out is not carried by them and gets no lanes.

# %%
carried = sorted({lane.word_id for lane in new_lanes})
print(f"words carried by the new bus: {carried}")
assert carried == list(zone.words_with_site_buses)

sample = min(
    (lane for lane in new_lanes if lane.direction == Direction.FORWARD),
    key=lambda lane: lane.word_id,
)
print(f"example routed lane (forward): word {sample.word_id}, site {sample.site_id}")
for point in extended.paths[sample]:
    print(f"    {point[0]:8.1f}, {point[1]:6.1f} um")

# %% [markdown]
# ## What the builder refuses
#
# Three mistakes are caught at the call that makes them, rather than
# surfacing later as a warning or a silently wrong spec.

# %%
# 1. A bus that repeats an existing one. A bus's index is its bus_id and
#    preserved paths key off it, so a duplicate only adds redundant lanes.
try:
    builder.add_site_bus(zone.name, src=[0], dst=[7])
except ValueError as error:
    print(f"duplicate rejected: {error}")

# 2. A move no AOD can perform. Reversing the order of the sites would
#    require two tones to pass through each other.
try:
    builder.add_site_bus(zone.name, src=[0, 1, 2, 3], dst=[7, 6, 5, 4])
except ValueError as error:
    print(f"\nunperformable rejected: {error}")

# 3. Adding a bus with no clearances to route it. Without them the search
#    cannot run, and the builder says so rather than guessing.
try:
    no_clearances = ArchBuilder.from_spec(spec)
    no_clearances.add_site_bus(zone.name, src=[0], dst=[7])
    no_clearances.build()
except ValueError as error:
    print(f"\nunroutable rejected: {error}")

# %% [markdown]
# ## Rebuilding without extending changes nothing at all
#
# A plain round-trip is lossless, paths included — worth knowing if you
# want to inspect an architecture through the builder without altering it.

# %%
untouched = ArchBuilder.from_spec(spec).build(
    feed_forward=spec.feed_forward, atom_reloading=spec.atom_reloading
)
assert untouched._inner.words == spec._inner.words
assert list(untouched._inner.zones) == list(spec._inner.zones)
assert untouched.paths == spec.paths
print("round-trip without changes: spec reproduced exactly")
