from bloqade import stim


def test_wrapper_x():

    @stim.main
    def main_x():
        stim.X(targets=(0, 1, 2), dagger=False)

    @stim.main
    def main_x_wrap():
        stim.x(targets=(0, 1, 2), dagger=False)

    assert main_x.callable_region.is_structurally_equal(main_x_wrap.callable_region)


def test_wrapper_y():

    @stim.main
    def main_y():
        stim.Y(targets=(0, 1, 2), dagger=False)

    @stim.main
    def main_y_wrap():
        stim.y(targets=(0, 1, 2), dagger=False)

    assert main_y.callable_region.is_structurally_equal(main_y_wrap.callable_region)


def test_wrapper_z():

    @stim.main
    def main_z():
        stim.Z(targets=(0, 1, 2), dagger=False)

    @stim.main
    def main_z_wrap():
        stim.z(targets=(0, 1, 2), dagger=False)

    assert main_z.callable_region.is_structurally_equal(main_z_wrap.callable_region)


def test_wrapper_identity():

    @stim.main
    def main_identity():
        stim.Identity(targets=(0, 1, 2), dagger=False)

    @stim.main
    def main_identity_wrap():
        stim.identity(targets=(0, 1, 2), dagger=False)

    assert main_identity.callable_region.is_structurally_equal(
        main_identity_wrap.callable_region
    )


def test_wrapper_h():

    @stim.main
    def main_h():
        stim.H(targets=(0, 1, 2), dagger=False)

    @stim.main
    def main_h_wrap():
        stim.h(targets=(0, 1, 2), dagger=False)

    assert main_h.callable_region.is_structurally_equal(main_h_wrap.callable_region)


def test_wrapper_s():

    @stim.main
    def main_s():
        stim.S(targets=(0, 1, 2), dagger=False)

    @stim.main
    def main_s_wrap():
        stim.s(targets=(0, 1, 2), dagger=False)

    assert main_s.callable_region.is_structurally_equal(main_s_wrap.callable_region)


def test_wrapper_sqrt_x():

    @stim.main
    def main_sqrt_x():
        stim.SqrtX(targets=(0, 1, 2), dagger=False)

    @stim.main
    def main_sqrt_x_wrap():
        stim.sqrt_x(targets=(0, 1, 2), dagger=False)

    assert main_sqrt_x.callable_region.is_structurally_equal(
        main_sqrt_x_wrap.callable_region
    )


def test_wrapper_sqrt_y():

    @stim.main
    def main_sqrt_y():
        stim.SqrtY(targets=(0, 1, 2), dagger=False)

    @stim.main
    def main_sqrt_y_wrap():
        stim.sqrt_y(targets=(0, 1, 2), dagger=False)

    assert main_sqrt_y.callable_region.is_structurally_equal(
        main_sqrt_y_wrap.callable_region
    )


def test_wrapper_sqrt_z():

    @stim.main
    def main_sqrt_z():
        stim.SqrtZ(targets=(0, 1, 2), dagger=False)

    @stim.main
    def main_sqrt_z_wrap():
        stim.sqrt_z(targets=(0, 1, 2), dagger=False)

    assert main_sqrt_z.callable_region.is_structurally_equal(
        main_sqrt_z_wrap.callable_region
    )


def test_wrapper_swap():

    @stim.main
    def main_swap():
        stim.Swap(controls=(0, 1, 2), targets=(3, 4, 5), dagger=False)

    @stim.main
    def main_swap_wrap():
        stim.swap(controls=(0, 1, 2), targets=(3, 4, 5), dagger=False)

    assert main_swap.callable_region.is_structurally_equal(
        main_swap_wrap.callable_region
    )


def test_wrapper_cx():

    @stim.main
    def main_cx():
        stim.CX(controls=(0, 1, 2), targets=(3, 4, 5), dagger=False)

    @stim.main
    def main_cx_wrap():
        stim.cx(controls=(0, 1, 2), targets=(3, 4, 5), dagger=False)

    assert main_cx.callable_region.is_structurally_equal(main_cx_wrap.callable_region)


def test_wrapper_cy():

    @stim.main
    def main_cy():
        stim.CY(controls=(0, 1, 2), targets=(3, 4, 5), dagger=False)

    @stim.main
    def main_cy_wrap():
        stim.cy(controls=(0, 1, 2), targets=(3, 4, 5), dagger=False)

    assert main_cy.callable_region.is_structurally_equal(main_cy_wrap.callable_region)


def test_wrapper_cz():

    @stim.main
    def main_cz():
        stim.CZ(controls=(0, 1, 2), targets=(3, 4, 5), dagger=False)

    @stim.main
    def main_cz_wrap():
        stim.cz(controls=(0, 1, 2), targets=(3, 4, 5), dagger=False)

    assert main_cz.callable_region.is_structurally_equal(main_cz_wrap.callable_region)


def test_wrap_pauli_string():

    @stim.main
    def main_pauli_string():
        stim.NewPauliString(
            string=("X", "Z", "Y"), flipped=(False, False, False), targets=(0, 1, 2)
        )

    @stim.main
    def main_pauli_string_wrap():
        stim.pauli_string(
            string=("X", "Z", "Y"), flipped=(False, False, False), targets=(0, 1, 2)
        )

    assert main_pauli_string.callable_region.is_structurally_equal(
        main_pauli_string_wrap.callable_region
    )


def test_wrap_spp():

    @stim.main
    def main_spp():
        stim.SPP(
            targets=(
                stim.pauli_string(
                    string=("X", "Z", "Y"),
                    flipped=(False, False, False),
                    targets=(0, 1, 2),
                ),
                stim.pauli_string(
                    string=("Z", "Z", "Z"),
                    flipped=(False, False, False),
                    targets=(3, 4, 5),
                ),
            ),
            dagger=False,
        )

    @stim.main
    def main_spp_wrap():
        stim.spp(
            targets=(
                stim.pauli_string(
                    string=("X", "Z", "Y"),
                    flipped=(False, False, False),
                    targets=(0, 1, 2),
                ),
                stim.pauli_string(
                    string=("Z", "Z", "Z"),
                    flipped=(False, False, False),
                    targets=(3, 4, 5),
                ),
            ),
            dagger=False,
        )

    assert main_spp.callable_region.is_structurally_equal(main_spp_wrap.callable_region)


def test_wrap_rec():

    @stim.main
    def main_rec():
        stim.GetRecord(0)

    @stim.main
    def main_rec_wrap():
        stim.rec(0)

    assert main_rec.callable_region.is_structurally_equal(main_rec_wrap.callable_region)


def test_wrap_detector():

    @stim.main
    def main_detector():
        stim.Detector(
            coord=(0.0, 1.0, 2.0), targets=(stim.rec(0), stim.rec(1), stim.rec(2))
        )

    @stim.main
    def main_detector_wrap():
        stim.detector(
            coord=(0.0, 1.0, 2.0), targets=(stim.rec(0), stim.rec(1), stim.rec(2))
        )

    assert main_detector.callable_region.is_structurally_equal(
        main_detector_wrap.callable_region
    )


def test_wrap_observable_include():

    @stim.main
    def main_observable_include():
        stim.ObservableInclude(idx=0, targets=(stim.rec(0), stim.rec(1), stim.rec(2)))

    @stim.main
    def main_observable_include_wrap():
        stim.observable_include(idx=0, targets=(stim.rec(0), stim.rec(1), stim.rec(2)))

    assert main_observable_include.callable_region.is_structurally_equal(
        main_observable_include_wrap.callable_region
    )


def test_wrap_tick():

    @stim.main
    def main_tick():
        stim.Tick()

    @stim.main
    def main_tick_wrap():
        stim.tick()

    assert main_tick.callable_region.is_structurally_equal(
        main_tick_wrap.callable_region
    )


def test_wrap_mz():

    @stim.main
    def main_mz():
        stim.MZ(p=0.5, targets=(0, 1, 2))

    @stim.main
    def main_mz_wrap():
        stim.mz(p=0.5, targets=(0, 1, 2))

    assert main_mz.callable_region.is_structurally_equal(main_mz_wrap.callable_region)


def test_wrap_my():

    @stim.main
    def main_my():
        stim.MY(p=0.5, targets=(0, 1, 2))

    @stim.main
    def main_my_wrap():
        stim.my(p=0.5, targets=(0, 1, 2))

    assert main_my.callable_region.is_structurally_equal(main_my_wrap.callable_region)


def test_wrap_mx():

    @stim.main
    def main_mx():
        stim.MX(p=0.5, targets=(0, 1, 2))

    @stim.main
    def main_mx_wrap():
        stim.mx(p=0.5, targets=(0, 1, 2))

    assert main_mx.callable_region.is_structurally_equal(main_mx_wrap.callable_region)


def test_wrap_mzz():

    @stim.main
    def main_mzz():
        stim.MZZ(p=0.5, targets=(0, 1, 2))

    @stim.main
    def main_mzz_wrap():
        stim.mzz(p=0.5, targets=(0, 1, 2))

    assert main_mzz.callable_region.is_structurally_equal(main_mzz_wrap.callable_region)


def test_wrap_myy():

    @stim.main
    def main_myy():
        stim.MYY(p=0.5, targets=(0, 1, 2))

    @stim.main
    def main_myy_wrap():
        stim.myy(p=0.5, targets=(0, 1, 2))

    assert main_myy.callable_region.is_structurally_equal(main_myy_wrap.callable_region)


def test_wrap_mxx():

    @stim.main
    def main_mxx():
        stim.MXX(p=0.5, targets=(0, 1, 2))

    @stim.main
    def main_mxx_wrap():
        stim.mxx(p=0.5, targets=(0, 1, 2))

    assert main_mxx.callable_region.is_structurally_equal(main_mxx_wrap.callable_region)


def test_wrap_mpp():

    @stim.main
    def main_mpp():
        stim.PPMeasurement(
            p=0.5,
            targets=(
                stim.pauli_string(
                    string=("X", "Z", "Y"),
                    flipped=(False, False, False),
                    targets=(0, 1, 2),
                ),
                stim.pauli_string(
                    string=("Z", "Z", "Z"),
                    flipped=(False, False, False),
                    targets=(3, 4, 5),
                ),
            ),
        )

    @stim.main
    def main_mpp_wrap():
        stim.mpp(
            p=0.5,
            targets=(
                stim.pauli_string(
                    string=("X", "Z", "Y"),
                    flipped=(False, False, False),
                    targets=(0, 1, 2),
                ),
                stim.pauli_string(
                    string=("Z", "Z", "Z"),
                    flipped=(False, False, False),
                    targets=(3, 4, 5),
                ),
            ),
        )

    assert main_mpp.callable_region.is_structurally_equal(main_mpp_wrap.callable_region)


def test_wrap_rz():

    @stim.main
    def main_rz():
        stim.RZ(targets=(0, 1, 2))

    @stim.main
    def main_rz_wrap():
        stim.rz(targets=(0, 1, 2))

    assert main_rz.callable_region.is_structurally_equal(main_rz_wrap.callable_region)


def test_wrap_rx():

    @stim.main
    def main_rx():
        stim.RX(targets=(0, 1, 2))

    @stim.main
    def main_rx_wrap():
        stim.rx(targets=(0, 1, 2))

    assert main_rx.callable_region.is_structurally_equal(main_rx_wrap.callable_region)


def test_wrap_ry():

    @stim.main
    def main_ry():
        stim.RY(targets=(0, 1, 2))

    @stim.main
    def main_ry_wrap():
        stim.ry(targets=(0, 1, 2))

    assert main_ry.callable_region.is_structurally_equal(main_ry_wrap.callable_region)
