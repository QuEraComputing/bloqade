from typing import Any

from kirin.dialects import ilist
from bloqade.lanes.dialects import steane713 as operations

from bloqade import squin, types
from bloqade.gemini.logical.group import kernel


@kernel(aggressive_unroll=True, verify=False)
def set_detector(meas: ilist.IList[types.MeasurementResult, Any]):
    return ilist.IList(
        [
            squin.set_detector(
                [meas[0], meas[1], meas[2], meas[3]], coordinates=[0, 0]
            ),
            squin.set_detector(
                [meas[1], meas[2], meas[4], meas[5]], coordinates=[0, 1]
            ),
            squin.set_detector(
                [meas[2], meas[3], meas[4], meas[6]], coordinates=[0, 2]
            ),
        ]
    )


@kernel(aggressive_unroll=True, verify=False)
def set_observable(meas: ilist.IList[types.MeasurementResult, Any]):
    return squin.set_observable([meas[0], meas[1], meas[5]])


@kernel(aggressive_unroll=True, verify=False)
def default_post_processing(register: ilist.IList[types.Qubit, Any]):
    measurements = operations.terminal_measure(register)

    detectors = set_detector(measurements[0])
    observables = ilist.IList([set_observable(measurements[0])])
    for i in range(1, len(register)):
        detectors = detectors + set_detector(measurements[i])
        observables = observables + [set_observable(measurements[i])]

    return detectors, observables
