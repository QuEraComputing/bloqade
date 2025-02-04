from bloqade import stim


def test_wrapper_x():

    @stim.main
    def main1():
        stim.X(targets=(0, 1, 2), dagger=False)

    @stim.main
    def main2():
        stim.x(targets=(0, 1, 2), dagger=False)

    assert main1.callable_region.is_structurally_equal(main2.callable_region)
