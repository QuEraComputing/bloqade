# Background

## Neutral Atom Qubits

A key feature of a quantum computer is the ability to physically represent qubits. In neutral atom computers, the qubit is represented in the electronic state of the valence electron of Rubdidium 87. Arrays of individual atoms are held by laser tweezers, and quantum computations are executed by manipulating the electronic state of each atom using lasers and RF fields. Entanglement can be generated using the [Rydberg state](https://en.wikipedia.org/wiki/Rydberg_atom), which is a highly excited state that strongly interacts with adjacent atoms through a $R^{-6}$ power law Van der Waals force.



## Analog mode Quantum Computing

There are two modes of quantum computation that [neutral atoms](#neutral-atom-qubits) are capable of: [*Analog*](#analog-mode) and [*Digital*](#digital-mode). In analog mode, the qubit is represented as in a ground state and a Rydberg state of an atom. The atoms are placed in user-specified arbitrary positions in a 2d space, and quantum computations can be enacted by driving the atoms between the ground and Rydberg state. However, adjacent atoms in the Rydberg state are always interacting, so the computation is done through a time evolution of the atoms via the Schrodinger equation

$$
i \hbar \dfrac{\partial}{\partial t} | \psi \rangle = \hat{\mathcal{H}}(t) | \psi \rangle,  \\
$$

Were $H$ is a time-dependent "Rydberg atom" Hamiltonian.

$$
\frac{\mathcal{H}(t)}{\hbar} = \sum_j \frac{\Omega_j(t)}{2} \left( e^{i \phi_j(t) } | g_j \rangle  \langle r_j | + e^{-i \phi_j(t) } | r_j \rangle  \langle g_j | \right) - \sum_j \Delta_j(t) \hat{n}_j + \sum_{j < k} V_{jk} \hat{n}_j \hat{n}_k,
$$

where: $\Omega_j$, $\phi_j$, and $\Delta_j$ denote the Rabi frequency *amplitude*, laser *phase*, and the *detuning* of the driving laser field on atom (qubit) $j$ coupling the two states  $| g_j \rangle$ (ground state) and $| r_j \rangle$ (Rydberg state); $\hat{n}_j = |r_j\rangle \langle r_j|$ is the number operator, and $V_{jk} = C_6/|\mathbf{x}_j - \mathbf{x}_k|^6$ describes the Rydberg interaction (van der Waals interaction) between atoms $j$ and $k$ where $\mathbf{x}_j$ denotes the *position* of the atom $j$; $C_6$ is the Rydberg interaction constant that depends on the particular Rydberg state used. For Bloqade, the default $C_6 = 862690 \times 2\pi \text{ MHz μm}^6$ for $|r \rangle = \lvert 70S_{1/2} \rangle$ of the $^{87}$Rb atoms; $\hbar$ is the reduced Planck's constant.


For a more nuanced read about the neutral atoms that Bloqade and *Aquila* use, refer to QuEra's qBook section on [Qubits by puffing up atoms](https://qbook.quera.com/learn/?course=6630211af30e7d0013c66147&file=6630211af30e7d0013c66149).

You can find a brief explanation of the distinction below but for a more in-depth explanation you can refer to QuEra's qBook section on [Analog vs Digital Quantum Computing](https://qbook.quera.com/learn/?course=6630211af30e7d0013c66147&file=6630211af30e7d0013c6614a). For more details on QuEra's cloud-accessible analog mode computer Aquila, please check out the [Aquila whitepaper](https://arxiv.org/abs/2306.11727).

### Digital Mode

In the Digital Mode individual or multiple groups of qubits are controlled by applying *gates* (individual unitary operations). The digital mode qubit is represented in the two hyperfine clock ground states of the Rubidium 87 atom. These two states are extremely weakly interactive with the environment and other adjacent atoms, which leads to a very long coherence time upwards of 1 second. Single-qubit gates can be executed through a Raman laser drive coupling the two states to enact arbitrary rotations.

Unlike Analog mode where the Rydberg state is persistent as part of the qubit encoding into the electronic states, digital mode only temporarily excites the atoms to the Rydberg state in order to interact with adjacent qubits, a process which typically takes less than ~1usec. Thus, a neutral atom entangling gate is executed by bringing multiple atoms together within the Rydberg blockade radius, and then doing some time-dependent drive between the hyperfine ground states and the Rydberg state, so that the final state returns to the hyperfine ground states. Due to the Rydberg blockade, only one atom can be in the Rydberg state at a time, which creates entanglement between the atoms. For more details see this paper on a [recent demonstration of high fidelity gates](https://www.nature.com/articles/s41586-023-06481-y).

A unique advantage of reconfigurable neutral atom architectures is parallelism: the same laser can effect many lasers by aiming it in the same plane as the atom array. A single global Raman laser can enact the same parallel single-qubit gate on all qubits at the same time, and a single Rydberg laser (technically, two counter-propagating) can enact the same parallel multi-qubit gate on all cliques of qubits in an entangling region of the array. For more details see this paper on a [recent demonstration of reconfigurable architectures](https://www.nature.com/articles/s41586-023-06927-3). For this reason, it is important to represent quantum executions and circuits to be as parallel as possible. In our qasm2 dialect, we have extended qasm to natively include parallelism-- for example, `qasm2.parallel.cx(controls, targets)` represents a parallel CNOT gate between a list of `controls` on a list of `targets`.


### Reconfigurable architectures and "all to all" connectivity

A second advantage of reconfigurable neutral atom architectures is reconfigurability: atoms can be moved in parallel between sites in the array. QuEra's devices will have a *zoned architecture*, with distinct storage and entanglement zones and the ability to move atoms between them using a set of dynamic crossed AOD laser tweezers. This mobility can be considered as an *efficient parallel swap* gate, where any qubit can easily be moved to be adjacent to any other to enact entangling gates. For this reason, reconfigurable neutral atoms do not have a "connectivity graph" in the traditional sense-- instead, they have an "all-to-all" connectivity. There are still some technical constraints on this connectivity due to restrictions on the crossed AOD which we will detail when we open-source a move level dialect set in the near future.
