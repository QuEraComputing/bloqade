from kirin import ir
from bloqade import qasm2
from kirin.analysis import const
from kirin.dialects import py
from bloqade.analysis.address import (
    NotQubit,
    AddressReg,
    AnyAddress,
    AddressQubit,
    AddressTuple,
    AddressAnalysis,
)

constprop = const.Propagate(qasm2.main.add(py.tuple), save_all_ssa=True)
address = AddressAnalysis(qasm2.main.add(py.tuple), save_all_ssa=True)


def constprop_results(mt: ir.Method):
    constprop.results.clear()
    constprop.eval(mt, tuple(const.JointResult.top() for _ in mt.args))
    return constprop.results


def address_results(mt: ir.Method):
    address.clear()
    const_results = constprop_results(mt)
    # fixed_count.print(analysis=const_results)
    address.constprop_results = const_results
    return address.eval(mt, tuple(AddressTuple.top() for _ in mt.args)).expect()


def test_fixed_count():
    @qasm2.main
    def fixed_count():
        ra = qasm2.qreg(3)
        rb = qasm2.qreg(4)
        q0 = ra[0]
        q3 = rb[1]
        qasm2.X(q0)
        qasm2.X(q3)
        return q3

    result = address_results(fixed_count)
    # fixed_count.print(analysis=address.results)
    assert isinstance(result, AddressQubit)
    assert result.data == range(3, 7)[1]
    assert address.qubit_count == 7


def test_multiple_return_only_reg():

    @qasm2.main.add(py.tuple)
    def tuple_count():
        ra = qasm2.qreg(3)
        rb = qasm2.qreg(4)
        return ra, rb

    # tuple_count.dce()
    result = address_results(tuple_count)
    tuple_count.code.print(analysis=address.results)
    assert isinstance(result, AddressTuple)
    assert isinstance(result.data[0], AddressReg) and result.data[0].data == range(0, 3)
    assert isinstance(result.data[1], AddressReg) and result.data[1].data == range(3, 7)


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
    result = address_results(dynamic_address)
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
    result = address_results(multi_return_cnt)
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
    result = address_results(list_count_analy)
    print(result)
    assert isinstance(result, AddressTuple)
    assert isinstance(result.data[0], AddressQubit) and result.data[0].data == 0
    assert isinstance(result.data[1], AddressQubit) and result.data[1].data == 1
    assert isinstance(result.data[2], AddressQubit) and result.data[2].data == 3


def test_tuple_qubits():
    @qasm2.main.add(py.tuple)
    def list_count_analy2():
        ra = qasm2.qreg(3)
        rb = qasm2.qreg(4)

        f = (ra[0], ra[1], rb[0])
        return f

    list_count_analy2.code.print()
    result = address_results(list_count_analy2)
    assert isinstance(result, AddressTuple)
    assert isinstance(result.data[0], AddressQubit) and result.data[0].data == 0
    assert isinstance(result.data[1], AddressQubit) and result.data[1].data == 1
    assert isinstance(result.data[2], AddressQubit) and result.data[2].data == 3


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
    result = address_results(test_alias)
    assert isinstance(result, AddressQubit)
    assert result.data == 0
