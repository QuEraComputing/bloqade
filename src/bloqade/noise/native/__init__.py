"""NOTE: This module is not guaranteed to be supported long-term in bloqade. We will be
moving towards a more general approach to noise modeling in the future. """

from .model import (
    NoiseModelABC as NoiseModelABC,
    GateNoiseParams as GateNoiseParams,
    TwoRowZoneModel as TwoRowZoneModel,
)
from .stmts import (
    PauliChannel as PauliChannel,
    CZPauliChannel as CZPauliChannel,
    AtomLossChannel as AtomLossChannel,
)
from .rewrite import RemoveNoisePass as RemoveNoisePass
from ._dialect import dialect as dialect
from ._wrappers import (
    pauli_channel as pauli_channel,
    cz_pauli_channel as cz_pauli_channel,
    atom_loss_channel as atom_loss_channel,
)
