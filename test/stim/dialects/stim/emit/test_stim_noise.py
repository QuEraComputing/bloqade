from bloqade import stim

from .base import codegen


def test_noise():

    @stim.main
    def test_pauli2():
        stim.PauliChannel2(
            pix=0.1,
            piy=0.1,
            piz=0.1,
            pxi=0.1,
            pxx=0.1,
            pxy=0.1,
            pxz=0.1,
            pyi=0.1,
            pyx=0.1,
            pyy=0.1,
            pyz=0.1,
            pzi=0.1,
            pzx=0.1,
            pzy=0.1,
            pzz=0.1,
            targets=(0, 3, 4, 5),
        )

    out = codegen(test_pauli2)
    assert (
        out.strip()
        == "PAULI_CHANNEL_2(0.10000000, 0.10000000, 0.10000000, 0.10000000, 0.10000000, 0.10000000, 0.10000000, 0.10000000, 0.10000000, 0.10000000, 0.10000000, 0.10000000, 0.10000000, 0.10000000, 0.10000000) 0 3 4 5"
    )


test_noise()
