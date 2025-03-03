---
date: 2025-03-01
authors:
    - jwurtz
    - rogerluo
    - kaihsin
    - weinbe58
    - johnzl-777
---
# A new journey for Bloqade

In 2023 we were excited to introduce bloqade, a python SDK for programming and interfacing with analog mode neutral atom hardware. Today, we introduce the next generation of Bloqade: as well as programming analog-mode computation, our new bloqade-circuits module enables programming digital and gate-based computation, with an eye on near-term NISQ demonstrations and intermediate-term fault tolerant solutions. Don’t worry; all of your favorite features of the previous generation of bloqade are still there under the `bloqade.analog` namespace, but now you can explore digital-mode computation specialized to reconfigurable neutral atoms.
Why have we built this new module? There are plenty of incredible quantum programming packages, such as [qiskit]( https://www.ibm.com/quantum/qiskit) and [cirq]( https://quantumai.google/cirq), as well as an entire ecosystem of middleware providers with specialized pipelines to turn abstract problems into circuits. However, these packages may not be everything that is needed for efficient hardware execution on neutral atom hardware: **a circuits-only representation of quantum executions may be the incorrect abstraction for effective hardware-level programs**. This is a challenge: we want to enable everyone to maximally leverage the power of neutral atom quantum computers beyond abstract circuit representations. For this reason, we are building Bloqade to be a hardware-oriented SDK to represent hybrid executions on reconfigurable neutral atom hardware. In this way, bloqade can be integrated into the larger ecosystem—for example, codegen of qasm from a bloqade-circuits program, but be an SDK specialized to our hardware: **THE SDK for neutral atoms**.

The vision of bloqade is to empower quantum scientists, from applications development to algorithmic co-design, to build hybrid quantum-classical programs that leverage the strength of neutral atom quantum computers and have a real chance of demonstrating quantum utility. Bloqade is built on top of [Kirin (no link)](), an open source compiler toolchain designed for kernel-level programs and composable representations.

## Composable quantum programming

As of today, bloqade has two components: bloqade-analog and bloqade-circuits. Bloqade-analog is the SDK for analog-mode neutral atom computers and includes several handy utilities ranging from building or analyzing analog programs, to emulation or executing on QuEra's cloud-accessible hardware "Aquila". Bloqade-circuits is the initial iteration to represent digital circuit execution using gate-based quantum computing on reconfigurable neutral atoms. It extends the QASM2 language to include extra annotation of circuits that is important for efficient execution, such as parallelism and global gates. As well as being able to construct quantum programs with the full convenience of typical classical programming-- such as loops and control flow—bloqade-circuits also includes basic compiler transformation passes, emulation, and code generation.

But bloqade is not done with just these two components. We envision adding new components (called "dialects") which help you write programs which are tuned for optimal performance in an error corrected era, and on neutral atom hardware. Stay tuned and help us build the future of quantum computing as we build out new components, such as QEC and atom moving dialects.


# Hardware-oriented programming and co-design
At its core, Bloqade strives to be the neutral atom SDK for getting the most out of today's and tomorrows' quantum hardware. It is clear that the circuit-level abstraction is not enough to program real quantum hardware; indeed, tomorrows' quantum demonstrations and applications must program at the hardware level and develop special tooling to compile higher-level abstractions to efficient implementations. We call this process **"co-design"**: designing algorithms specialized to near-term hardware, with an eye on nontrivial demonstrations and scalable solutions. Ultimately, this co-design approach requires hardware-specific DSLs which explicitly represent the native executions on neutral atom hardware: in other words, Bloqade.


# Hybrid computing beyond circuits
Many quantum algorithms are hybrid, requiring both classical and quantum resources to work together in a hybrid computation architecture. This could be anything from syndrome extraction and measurement-based computing to variational parameter updates in VQE methods and orbital fragmentation methods in molecular simulation. Through the use of the Kirin compiler infrastructure, Bloqade embraces this philosophy of heterogeneous compute. Kirin programs are written as (compositions of) [kernels](https://en.wikipedia.org/wiki/Compute_kernel)-- subroutines that are intended to run on particular hardware (such as QPUs), or orchestrated to run on heterogeneous compute (such as a real-time classical runtime plus a QPU). These subroutines-- plus the built-in hybrid representations-- enable many key primitives, such as error correction.

Additionally, the ability to compose functions together and to use typical classical programming structures like `if` and `while` enables many simplifications in writing raw circuit executions. In fact, `while` loops and the ability to dynamically allocate new memory (which is not known until runtime) enables many powerful subroutines and is natively enabled with bloqade's kernel-based representation; for example, see [this implementation](digital/examples/repeat_until_success.py) of a repeat-until-success program.

# Analog, digital, logical: towards real quantum utility
The first step in Bloqade was building out the analog mode SDK, designed to interface with QuEra’s cloud-accessible analog-mode neutral-atom quantum computer Aquila, as well as enable analysis and scientific discovery in analog quantum computing. But the journey should not stop there: real quantum utility is error corrected and requires robust algorithmic exploration and design of quantum primitives, in-depth analysis of near-term hardware performance and benchmarking, and building pipelines and hybrid architectures that are intended not just for today’s demonstrators but also for tomorrow’s utility-scale hardware. By introducing the next generation of Bloqade, we hope to enable this exploration by adding in support for near-term digital and intermediate-term logical representations of hybrid quantum computations.

# Learn more
Bloqade is open-source project and can be freely downloaded; you can learn how to do so [here](install.md). If you want to see how to write programs with of the new bloqade-circuits package, check out our examples [here](digital/index.md). If you would like to learn more about QuEra computing, check out our [webpage](quera.com) or discover our many academic publications and demonstrations.
