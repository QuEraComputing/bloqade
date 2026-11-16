# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: bloqade-lanes (3.12.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial for Customizing Existing Architecture + Visualizing Architecture
# In this notebook, we go over how you might define a new architecture, visualize it, and add buses on top of an existing architecture.

# %% [markdown]
# ## Defining Architecture Layout
# First, we define a custom architecture where each word is an SLM site in the architecture. For the purposes of this tutorial, we can treat each SLM site as having an integer ID which we can later address.

from bloqade.lanes.visualize.arch import ArchVisualizer

# %%
from bloqade.gemini.arch.build.v2 import ArchBuilder

# %% [markdown]
# We can define some constants for the architecture layout and spacing for the generated lane paths.

# %%
# Define the number of rows and columns in the architecture.
NUM_ROWS = 5
NUM_COLS = 8

# Define the spacing between the rows and pairs of columns in the architecture.
ROW_SPACING = 10
COLUMN_SPACING = 10
BLOCKADE_RADIUS = 2

# Define variables for how close the generated path is from each SLM site.
X_CLEARANCE = 0.5
Y_CLEARANCE = 1.0

# %% [markdown]
# The central object we use when building an architecture is an `ArchBuilder`. An `ArchBuilder` encapsulates the current state of the architecture and provides methods for adding SLM sites or buses.
#
# When defining an `ArchBuilder`, we need to supply two pieces of information:
# - `grid_shape`, which tells us how many rows/columns are in the architecture;
# - `word_shape`, which tells us the shape of each word in terms of the `(rows, cols)` it spans.

# %%
arch_builder = ArchBuilder(grid_shape=(NUM_ROWS, NUM_COLS), word_shape=(1, 1))

# %% [markdown]
# On our architecture, which is defined as a 5 by 8 grid, we can define what SLM sites are a part of what "word" in the architecture. This will help us create an addressing mechanism for each site in the architecture.
# > For the purposes of this example, we can address every site with a unique integer ID in row-major order starting from the word at position (0, 0) in the architecture.

# %%
for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        arch_builder.add_word([row], [col])

# %% [markdown]
# In general, our architectures can have multiple zones. For this notebook, we add a zone to the architecture builder named `gate`, and specify the positions of the rows and columns that are part of the zone via lists.
# > To compute the paths that each bus takes, we can specify the `x_clearance` and `y_clearance` parameters that indicate how far from SLM sites that each atom move path must be. These are used to compute the atom move paths.
#
# > We can also specify what words have site buses (in this case, we only have word buses in the architecture as every site is a word). As every word is at site index 0, we want `sites_with_word_buses` to contain only 0 (so that the specified word ID's have a bus between them).

# %%
arch_builder.add_zone(
    name="gate",
    rows=list(range(0, NUM_ROWS * ROW_SPACING, ROW_SPACING)),
    columns=[
        (
            i * (COLUMN_SPACING / 2.0)
            if i % 2 == 0
            else (i - 1) * (COLUMN_SPACING / 2.0) + 2.0
        )
        for i in range(NUM_COLS)
    ],
    x_clearance=X_CLEARANCE,
    y_clearance=Y_CLEARANCE,
    words_with_site_buses=[],
    sites_with_word_buses=[0],
)

# %% [markdown]
# We can additionally set the blockade radius for our architecture, which provides us a way to validate that two atoms are in blockade radius during compilation.

# %%
arch_builder.set_blockade_radius(BLOCKADE_RADIUS)

# %% [markdown]
# Finally, we can build the resulting architecture specification by calling the `build()` on the architecture builder.

# %%
arch_spec = arch_builder.build()

# %% [markdown]
# ## Visualizing the Architecture
# We can subsequently visualize our architecture given all of our choices above. To visualize the architecture, we can use the `ArchVisualizer` class that takes in an architecture, and has a method `plot_interactive` for visualizing the architecture in a notebook.

# %%
ArchVisualizer(arch_spec).plot_interactive()

# %% [markdown]
# You can see a variety of features in the above visualizer tool.
#
# Starting from the top left, you see that we have these `Labels off` and `Labels on` buttons, which can be used to toggle the display of the `(zone_id, word_id, site_id)` address for each atom. For the purposes of this tutorial, the `zone_id` and `site_id` are always 0 (because we have one zone, and one word per site, respectively), so the address is `(0, word_id, 0)`, where the `word_id` uniquely addresses the site.
#
# Visually, we can also verify that we have 5 rows and 8 columns in our architecture.

# %% [markdown]
# ## Customizing An Existing Architecture
# If we already have an `ArchSpec` defined, and want to customize it by adding buses to the architecture, we can do so using the `ArchBuilder.from_spec` method.
#
# > Note that if you want to add buses, you must also supply the `x_clearance` and `y_clearance` variables, which are used to compute the movement paths for each atom.

# %%
new_builder = ArchBuilder.from_spec(
    arch_spec, x_clearance=X_CLEARANCE, y_clearance=Y_CLEARANCE
)

# %% [markdown]
# If we wanted to add a bus between two words, we can do so by passing in the word ID's as a list of integers to the `add_word_bus` method.
#
# In this case, if we only wanted to add a bus between words 0 and 3, we can do so with the below function call.

# %%
new_builder.add_word_bus("gate", [0], [3])

# %% [markdown]
# We can subsequently obtain our new architecture spec via `new_builder.build()`.

# %%
new_spec = new_builder.build()

# %% [markdown]
# In our below architecture visualizer, we can see that we've added a word bus if we go to the right, and click on the dropdown for "Word Buses".
#
# We'll see a newly added word bus with the name `ID 0 · zone 0 · word bus 0`. Hovering over that bus will show us a path from the site with word ID 0 to word ID 3, which is what we expected.
#
# In such a way, by adding buses that respect the AOD constraints, we can in principle define a bus from any SLM site to any other SLM site.

# %%
ArchVisualizer(new_spec).plot_interactive()

# %% [markdown]
# ## Adding Buses to Respect AOD Constraints
# When adding buses to an architecture, it's important to keep in mind the AOD constraints that make a bus valid. Below, we will go through examples of invalid ways of adding buses to an architecture.
#
# > We will also demonstrate a convenience method for getting words by rows/columns, by indexing the `builder.words` like a 2D numpy array.

# %%
new_builder_aodconstr = ArchBuilder.from_spec(
    new_spec, x_clearance=X_CLEARANCE, y_clearance=Y_CLEARANCE
)

# %% [markdown]
# ### Case 1: No-Crossing Constraint
# Due to the no-crossing constraint on AOD's, the below word bus is invalid, as the vertical AOD's would have to cross to perform this move.

# %%
try:
    new_builder_aodconstr.add_word_bus(
        "gate",
        new_builder_aodconstr.words[0, 0] + new_builder_aodconstr.words[0, 2],
        new_builder_aodconstr.words[0, 5] + new_builder_aodconstr.words[0, 3],
    )
except ValueError as error:
    print(f"No crossing constraint: {error}")

# %% [markdown]
# ### Case 2: AOD Must Form Rectangle
# If we try to add a bus where we don't include sites that form a valid AOD rectangle, then we will also error. This means that, if we select sites at certain rows and columns, then we must include all sites at the intersection of the included rows and columns.
#
# In the below example, we attempt to pick up an atom at row 0, column 0 and row 1, column 2. However, because we do not also include the atoms at row 0, column 2 and row 1, column 0 in this bus, this is not a valid bus as those sites must be included.

# %%
try:
    new_builder_aodconstr.add_word_bus(
        "gate",
        new_builder_aodconstr.words[0, 0] + new_builder_aodconstr.words[1, 2],
        new_builder_aodconstr.words[0, 3] + new_builder_aodconstr.words[1, 5],
    )
except ValueError as error:
    print(f"AOD must form valid rectangle: {error}")

# %% [markdown]
# ### Case 3: Shape of Destination Differs from Source
# Another error case is that the shape of the tones at the source and destination differs.
#

# %%
try:
    new_builder_aodconstr.add_word_bus(
        "gate",
        new_builder_aodconstr.words[0, 0] + new_builder_aodconstr.words[0, 2],
        new_builder_aodconstr.words[0, 3] + new_builder_aodconstr.words[1, 5],
    )
except ValueError as error:
    print(f"Shape of destination differs from source: {error}")

# %%
