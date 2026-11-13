from bloqade.squin.stdlib import simple as squin_simple, broadcast as squin_broadcast

from bloqade.gemini import logical

SUPPORTED_GATE_EXPORTS = (
    "h",
    "s",
    "t",
    "x",
    "y",
    "z",
    "cx",
    "cy",
    "cz",
    "rx",
    "ry",
    "rz",
    "u3",
    "cnot",
    "swap",
    "s_adj",
    "shift",
    "t_adj",
    "sqrt_x",
    "sqrt_y",
    "sqrt_z",
    "sqrt_x_adj",
    "sqrt_y_adj",
    "sqrt_z_adj",
)


def test_logical_gate_exports_match_supported_squin_gates():
    for name in SUPPORTED_GATE_EXPORTS:
        assert getattr(logical, name) is getattr(squin_simple, name)
        assert getattr(logical.broadcast, name) is getattr(squin_broadcast, name)

    for name in ("ccz", "phased_xz"):
        assert not hasattr(logical, name)
        assert not hasattr(logical.broadcast, name)
