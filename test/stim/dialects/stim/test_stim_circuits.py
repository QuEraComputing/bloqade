from bloqade import stim
from bloqade.stim.emit import EmitStimMain

interp = EmitStimMain(stim.main)

def test_gates():
    @stim.main
    def test_single_qubit_gates():
        stim.sqrt_z(targets=(0, 1, 2), dagger=False)
        stim.x(targets=(0, 1, 2))
        stim.y(targets=(0, 1), dagger=True)
        stim.z(targets=(1, 2))
        stim.h(targets=(0, 1, 2), dagger=True)
        stim.s(targets=(0, 1, 2), dagger=False)
        stim.s(targets=(0, 1, 2), dagger=True)

    interp.run(test_single_qubit_gates, args=())
    print(interp.get_output())


    @stim.main
    def test_two_qubit_gates():
        stim.swap(targets=(2, 3))

    interp.run(test_two_qubit_gates, args=())
    print(interp.get_output())


    @stim.main
    def test_controlled_two_qubit_gates():
        stim.cx(controls=(0, 1), targets=(2, 3))
        stim.cy(controls=(0, 1), targets=(2, 3), dagger=True)
        stim.cz(controls=(0, 1), targets=(2, 3))

    interp.run(test_controlled_two_qubit_gates, args=())
    print(interp.get_output())


    # @stim.main
    # def test_spp():
    #     pauli_string = stim.PauliString(string=('X', 'Y', 'Z'), targets=(0, 1, 2), flipped=(True, False, True))
    #     stim.spp(targets=(pauli_string, ), dagger=True)

    # interp.run(test_spp, args=())
    # print(interp.get_output())


def test_noise():
    @stim.main
    def test_depolarize():
        stim.depolarize1(p=0.1, targets=(0, 1, 2))
        stim.depolarize2(p=0.1, targets=(0, 1))

    interp.run(test_depolarize, args=())
    print(interp.get_output())


    @stim.main
    def test_pauli_channel():
        stim.pauli_channel1(px=0.01, py=0.01, pz=0.1, targets=(0, 1, 2))
        stim.pauli_channel2(pix=0.01, piy=0.01, piz=0.1, 
                            pxi=0.01, pxx=0.01, pxy=0.01, pxz=0.1,
                            pyi=0.01, pyx=0.01, pyy=0.01, pyz=0.1,
                            pzi=0.1, pzx=0.1, pzy=0.1, pzz=0.2,
                            targets=(0, 1, 2, 3))
    
    interp.run(test_pauli_channel, args=())
    print(interp.get_output())


    @stim.main
    def test_pauli_error():
        stim.x_error(p=0.1, targets=(0, 1, 2))
        stim.y_error(p=0.1, targets=(0, 1))
        stim.z_error(p=0.1, targets=(1, 2))

    interp.run(test_pauli_error, args=())
    print(interp.get_output())
