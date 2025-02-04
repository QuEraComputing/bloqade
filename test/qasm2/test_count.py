from kirin import passes
from bloqade import qasm2
from kirin.dialects import py
from bloqade.analysis.address import (
    NotQubit,
    AddressReg,
    AnyAddress,
    AddressQubit,
    AddressTuple,
    AddressAnalysis,
)

address = AddressAnalysis(qasm2.main.add(py.tuple))
fold = passes.Fold(qasm2.main.add(py.tuple))


def test_fixed_count():
    @qasm2.main
    def fixed_count():
        ra = qasm2.qreg(3)
        rb = qasm2.qreg(4)
        q0 = ra[0]
        q3 = rb[1]
        qasm2.x(q0)
        qasm2.x(q3)
        return q3

    fold(fixed_count)
    results, ret = address.run_analysis(fixed_count)
    # fixed_count.print(analysis=address.results)
    assert isinstance(ret, AddressQubit)
    assert ret.data == range(3, 7)[1]
    assert address.qubit_count == 7


def test_multiple_return_only_reg():

    @qasm2.main.add(py.tuple)
    def tuple_count():
        ra = qasm2.qreg(3)
        rb = qasm2.qreg(4)
        return ra, rb

    # tuple_count.dce()
    fold(tuple_count)
    results, ret = address.run_analysis(tuple_count)
    tuple_count.code.print(analysis=results)
    assert isinstance(ret, AddressTuple)
    assert isinstance(ret.data[0], AddressReg) and ret.data[0].data == range(0, 3)
    assert isinstance(ret.data[1], AddressReg) and ret.data[1].data == range(3, 7)


def test_dynamic_address():
    @qasm2.main
    def dynamic_address():
        ra = qasm2.qreg(3)
        rb = qasm2.qreg(4)
        ca = qasm2.creg(2)
        qasm2.measure(ra[0], ca[0])
        qasm2.measure(rb[1], ca[1])
        if ca[0] == ca[1]:
            return ra
        else:
            return rb

    # dynamic_address.code.print()
    fold(dynamic_address)
    _, result = address.run_analysis(dynamic_address)
    assert isinstance(result, AnyAddress)


# NOTE: this is invalid for QASM2
# def test_cond_count2():
#     @qasm2.main
#     def cond_count2():
#         ra = qasm2.qreg(3)
#         rb = qasm2.qreg(4)
#         if 4 > 33:
#             return 3.0
#         else:
#             return 2.0

#     cond_count2.code.print()
#     result = address_results(cond_count2)
#     assert isinstance(result, NotQubit)


def test_multi_return():
    @qasm2.main.add(py.tuple)
    def multi_return_cnt():
        ra = qasm2.qreg(3)
        rb = qasm2.qreg(4)
        return ra, 3.0, rb

    multi_return_cnt.code.print()
    fold(multi_return_cnt)
    _, result = address.run_analysis(multi_return_cnt)
    print(result)
    assert isinstance(result, AddressTuple)
    assert isinstance(result.data[0], AddressReg)
    assert isinstance(result.data[1], NotQubit)
    assert isinstance(result.data[2], AddressReg)


def test_list():
    @qasm2.main
    def list_count_analy():
        ra = qasm2.qreg(3)
        rb = qasm2.qreg(4)
        f = [ra[0], ra[1], rb[0]]
        return f

    list_count_analy.code.print()
    _, ret = address.run_analysis(list_count_analy)
    print(ret)
    assert isinstance(ret, AddressTuple)
    assert isinstance(ret.data[0], AddressQubit) and ret.data[0].data == 0
    assert isinstance(ret.data[1], AddressQubit) and ret.data[1].data == 1
    assert isinstance(ret.data[2], AddressQubit) and ret.data[2].data == 3


def test_tuple_qubits():
    @qasm2.main.add(py.tuple)
    def list_count_analy2():
        ra = qasm2.qreg(3)
        rb = qasm2.qreg(4)

        f = (ra[0], ra[1], rb[0])
        return f

    list_count_analy2.code.print()
    _, ret = address.run_analysis(list_count_analy2)
    assert isinstance(ret, AddressTuple)
    assert isinstance(ret.data[0], AddressQubit) and ret.data[0].data == 0
    assert isinstance(ret.data[1], AddressQubit) and ret.data[1].data == 1
    assert isinstance(ret.data[2], AddressQubit) and ret.data[2].data == 3


# NOTE: invalid QASM2 program, use this test for future
# def test_tuple_qubits_tuple_add():
#     @qasm2.main.add(py.tuple)
#     def list_count_analy3():
#         ra = qasm2.qreg(3)
#         rb = qasm2.qreg(4)

#         f = (ra[0], ra[1], rb[0])
#         g = (ra[1], ra[0], rb[3])
#         return f + g

#     result = address_results(list_count_analy3)
#     list_count_analy3.print(analysis=address.results)
#     assert isinstance(result, AddressTuple)
#     assert len(result.data) == 6
#     assert isinstance(result.data[0], AddressQubit) and result.data[0].data == 0
#     assert isinstance(result.data[1], AddressQubit) and result.data[1].data == 1
#     assert isinstance(result.data[2], AddressQubit) and result.data[2].data == 3
#     assert isinstance(result.data[3], AddressQubit) and result.data[3].data == 1
#     assert isinstance(result.data[4], AddressQubit) and result.data[4].data == 0
#     assert isinstance(result.data[5], AddressQubit) and result.data[5].data == 6


def test_alias():

    @qasm2.main
    def test_alias():
        ra = qasm2.qreg(3)

        f = ra[0]
        g = f
        h = g

        return h

    test_alias.code.print()
    fold(test_alias)
    _, ret = address.run_analysis(test_alias)
    assert isinstance(ret, AddressQubit)
    assert ret.data == 0
