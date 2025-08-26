---
date: 2025-07-30
authors:
  - tcochran
  - lmartinez
  - dplankensteiner
---

# Simulating noisy circuits for near-term quantum hardware

We should change the name of this file eventually, but I'm excited about the final post of 2025: "Return of the Gemini."

## Motivate: Gemini-class digital QPUs (Luis)

Operating in an analog mode of quantum computation has opened to us exciting opportunities to leverage the flexibility of a neutral atom platform in exploring the application forefront (such as optimization problems, and machine learning), as well as in addressing more scientifically oriented questions (preparation of exotic phases of matter).

In our journey towards building uselful quantum computers, however, we find that operating in an analog mode we are rather limited on the range of problems that we can address, as we only have control on a handful of parameters of the underlying Hamiltonian of the quantum device. Here is where the need for a fully programable digital quantum computer comes into the picture. The promise of this mode of operation is the ability to encode non-native problems to the neutral atom platform. For instance, one of the envisioned and most exciting applications of quantum computers is the accurate simulation (in terms of estimating ground state energy) of electrons in molecules and materials. The fermionic statistics of the target particles to simulate contrast with the bosonic nature of the Rubidium atoms that constitute the building block of our platform. 

Furthermore, we need full control in the quantum device to encode interactions and tunneling parameters between the fermionic modes that discretize our target molecules and materials.
In our quest to reach this level of maturity in quantum hardware, we introduce Gemini-class devices, that incorporate digital programmability features in our neutral atom quantum computing platform.

## Circuit-level compared to hardware-level programming (David)

Gemini class devices are digital quantum computers.
This allows you to work on the circuit-level of abstraction rather than the hardware-level.
While this is certainly useful, in the current era of noisy intermediate scale quantum devices, you inevitably have to
consider potential noise processes when developing quantum programs.

When writing a circuit, noise processes can be taken into account as channels that cause decoherence thereby reducing
the overall circuit fidelity.
If the fidelity is too low, the computation may contain errors.
Before executing a circuit, you need to know whether this circuit will actually lead to the desired results.
This is where emulation comes in, which, in order to faithfully represent the results you can expect, needs to account
for noise.

At the hardware level, you always need to work with (or around) the noise that is adherent to the hardware you are
programming on.
This comes at the loss of abstraction and subsequently high-level tooling.
At the same time, however, in enables you to devise your own strategies in order to suppress noise in your specific
application, which will oftentimes outperform today's compilers.
Here, we will focus on circuit level programming, but please refer
to [bloqade-shuttle](https://queracomputing.github.io/bloqade-shuttle/dev/) to learn more about our hardware-level
programming capabilities.

Even including noise channels, circuit-level programming remains abstract in that you do not have to consider the
specific hardware you are running the circuit on.
However, it is the very nature of the noise channels, where the details of the hardware come into play.
To know whether a circuit will execute with a sufficiently high fidelity, the noise parameters and channels need to
represent the infidelity of the gates executed on the particular hardware (in this case Gemini).

In order to provide users with the required set of tools, we have spent considerable time researching and implementing
an easy-to-use framework that allows you to include Gemini's particular noise processes in a high-level circuit.

## Heuristic approach to noise (Tyler)

The abstraction of noise to the circuit-level allows all the noise sources on the device to be conglomerated into
"effective" Pauli noise channels. The effective channels are heuristic in nature, designed to capture the average
behavior of atoms when the system performs certain circuit-level operations. The quantitative results are based on
benchmarking experiments on Gemini hardware. They also take into account the dynamical decoupling that is needed to
eliminate the effects of differential stark shifts during atom moves. All in all, the models can be expressed by six
main noise channels:

1. Global single qubit gate error
    - Depolarizing error applied to all qubits after a single qubit gate that is applied to all qubits in parallel.

![Global single qubit gate error](./global_single_qubit_gate_error.png)

1. Local single qubit gate error
    - Depolarizing error applied to gated single qubits after a single qubit gate that is applied to a subset of qubits.

![Local single qubit gate error](./local_single_qubit_gate_error.png)

1. CZ gate error
    - Pauli error channel that is biases towards phase errors applied to both qubits that are within the blockage radius
      during a Rydberg pulse.
    - Incorporates errors from the Rydberg pulse and dynamical decoupling.
1. Unpaired Rydberg error
    - Pauli error channel is biased towards phase errors applied to single qubits that experience a Rydberg pulse but do
      not have a partner qubit within the blockade radius.
    - Incorporates errors from the Rydberg pulse and dynamical decoupling.

![Errors due to Rydberg pulses](./rydberg_error.png)

1. Mover error
    - Pauli error channel that is also biased towards phase errors applied to qubits that must move during a circuit.
    - Incorporates errors from transferring atoms from fixed tweezer traps to dynamical traps for moves, dynamical
      decoupling, move errors, and idling errors.
1. Sitter error
    - Pauli error channel that is applied to atoms that are stationary while other atoms are moving.
    - Incorporates errors from dynamical decoupling and idling errors.

![Errors due to atom moves](./move_error.png)

## Example: Noise in the GHZ state

It is often most convenient to study an example in order to learn how to use a set of tools.
To this end, we included a tutorial that shows you how to annotate a GHZ preparation circuit using the different
heuristic noise models.
We'll discuss some underlying concepts and highlight interesting parts of the tutorial in this section.
If you want all the details, please find [the full example here](../../../digital/examples/noisy_ghz.py).

### Flow chart (Tyler)

The intended workflow for using the circuit-level noise models is de in the flow chart below. The first step for
a user interested in testing a specific quantum algorithm is to write an explicit circuit
in [cirq](https://quantumai.google/cirq). Once defined in `cirq`, the circuit operations can be visualized with
`print(circuit)`, which can be used to inspect the circuit after being passed through the transformers within `bloqade`.
Also, at any point, one may choose the simulate the result of the circuit with the tools provided by cirq.

By passing the circuit through the transformers in `parallelize.py`, within the `bloqade-circuit.cirq_utils` repository,
the circuit structure can be brought closer to optimal for the neutral atom platform. Then, by using the transformers
in `noise.transform.py`, Pauli error channels are locally added to the circuit. The function `transform_circuit` acts as 
a wrapper for the different noise models, which correspond to different modes of hardware operation.

Finally, we maintain interoperability between `squin` (`bloqade`'s circuit-level intermediate representation) and
`cirq`. `squin` allows for simulation with different backends, including `pyqrack`, while also allowing for lowering to
hardware-level programs for executions on Gemini.

![cirq_utils_flowchart](./flowchart.png)

### Annotated circuit (Luis)

In practice, our heuristic noise models are used to annotate circuits with incoherent channels, with a "coarse-grained"
awareness of hardware. As a simple example, lets consider the following that assumes noise annotation according to a
two-zone layout:

![Noise annotation example. Two zone model](../../../digital/examples/example_annotation_fixed.svg)

In our annotation scheme, assume that we are iterating sequentially over the layers of a quantum circuit, and we encounter the layer depicted in a) on which we will add the noise channels. One important assumption in our two zone model to keep in mind is that the qubits corresponding to the gates appearing in a given layer, will be treated as qubits sitting in the gate zone, and thus, the rest of the qubits are assumed to be in the storage zone. 

To capture the cost of moves that we need to pay to reach a given configuration, we need to take a closer look to the layer that precedes our current target layer. In b), we explicitly show the layer that was annotated with noise in the previous iteration, with its corresponding qubit spatial layout below it. The move cost to reach the target qubit configuration before qubit execution is shown in c), and noise annotation is carried according to three sequential stages: 1) qubits that need to be removed from the gate zone undergo move error (orange crosses), and the rest get sitter error, 2)similarly, qubits that need to be added to the gate zone get move error and the rest get sitter noise, and 3) additional move error to "pair up" qubits is added before gate execution (notice than more than one layer might be needed to account for this cost).

Finally, noise is annotated after gates, where it is assumed that entangling gates are executed in hardware before single qubit gates. In doing so, qubits that are not participant in the entangling gates receive unpaired cz error (green pentagons).

### GHZ data (David)

Now, let's look at some results of [the example](../../../digital/examples/noisy_ghz.py) that compares the different
noise processes.

The different noise models lead to overall different infidelities of the circuit:

![GHZ circuit fidelity with different noise models](../../../digital/examples/noisy_ghz_fidelity.svg)

As you'd expect, the general trend of the fidelity is that it decreases with a growing number of qubits.
Depending on how many qubits you need, you may want to run the above simulation in order to decide whether you'd want to
operate Gemini in a one-zone or a two-zone setup.

On a more abstract level, you may also want to optimize the circuit that you use to obtain the result you want.
The example uses a linear depth GHZ algorithm, which is arguably not the best choice to prepare a GHZ state.

Using our noise framework you can explore and analyze different strategies to find the one best suited for your
particular application.
