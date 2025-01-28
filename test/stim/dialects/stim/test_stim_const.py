from bloqade import stim


def test_const():

    @stim.main
    def test_const_int():
        return 3

    out = test_const_int()

    assert out == 3
    assert isinstance(out, int)


def test_float():

    @stim.main
    def test_const_float():
        return 3.0

    out = test_const_float()

    assert out == 3.0
    assert isinstance(out, float)


def test_get_rec():

    @stim.main
    def get_rec():
        return stim.GetRecord(id=-3)

    get_rec.print()

    out = get_rec()

    assert isinstance(out, stim.RecordResult)
    assert out.value == -3
