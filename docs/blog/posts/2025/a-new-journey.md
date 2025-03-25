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

In 2023 we were excited to introduce Bloqade, a python SDK for programming and interfacing with analog mode neutral atom hardware based off feedback from our community as well as a need to make conducting experiments on our hardware easier. Today, we introduce the next generation of Bloqade: as well as programming analog-mode computation, our new bloqade module enables programming gate-based computation, with an eye on near-term NISQ demonstrations and intermediate-term fault tolerant solutions. Don’t worry; all of your favorite features of the previous generation of Bloqade are still there under the `bloqade.analog` namespace, but now you can explore digital-mode computation specialized to reconfigurable neutral atom architectures.
Why have we built this new module? There are plenty of incredible quantum programming packages, such as [Qiskit]( https://www.ibm.com/quantum/qiskit) and [Cirq]( https://quantumai.google/cirq), as well as an entire ecosystem of middleware providers with specialized pipelines to turn abstract problems into circuits. However, these packages may not be everything that is needed for efficient hardware execution on neutral atom hardware: **a circuits-only representation of quantum executions may be an insufficient abstraction for effective hardware-level programs**. This is a challenge: we want to enable everyone to maximally leverage the power of neutral atom quantum computers beyond abstract circuit representations. For this reason, we are building Bloqade to be a hardware-oriented SDK to represent hybrid executions on reconfigurable neutral atom hardware. In this way, Bloqade can be integrated into the larger ecosystem—for example, [code generation](https://en.wikipedia.org/wiki/Code_generation_(compiler)) of QASM from a Bloqade program, but be an SDK specialized to our hardware: **THE SDK for neutral atoms**.

The vision of Bloqade is to empower quantum scientists, working on things ranging from applications development to algorithmic co-design, to build hybrid quantum-classical programs that leverage the strength of neutral atom quantum computers and have a real chance of demonstrating quantum utility. Bloqade is built on top of [Kirin](https://github.com/QuEraComputing/kirin), an open source compiler infrastructure designed for kernel functions and embedded Domain-Specific Language (eDSL) creation.

## Composable quantum programming

As of today, Bloqade has two objectives: digital and analog quantum computing. `bloqade-analog` is the SDK for analog-mode neutral atom computers and includes several handy utilities ranging from building and analyzing analog programs, to emulation and execution on QuEra's cloud-accessible hardware "Aquila". `bloqade` is the initial iteration to represent digital circuit execution using gate-based quantum computing on reconfigurable neutral atom architecture. It extends the QASM2 language to include extra annotation of circuits that is important for efficient execution, such as parallelism and global gates. As well as being able to construct quantum programs with the full convenience of features found in classical programming languages - such as loops, control flows and closures - `bloqade` also includes basic compiler transformation passes, emulation, and code generation.

But `bloqade` is not done with just these two components. We envision adding new components (called "dialects") which help you write programs which are tuned for optimal performance in an error corrected era of neutral atom hardware. Stay tuned and help us build the future of quantum computing as we build out new components targeting QEC and atom moving!


## Hardware-oriented programming and co-design

At its core, Bloqade strives to be the neutral atom SDK for getting the most out of today's and tomorrows' quantum hardware. It is clear that the circuit-level abstraction is not enough to program real quantum hardware; indeed, tomorrows' quantum demonstrations and applications must program at the hardware level and develop special tooling to compile higher-level abstractions to efficient implementations. We call this process **"co-design"**: designing algorithms specialized to near-term hardware, with an eye on nontrivial demonstrations and scalable solutions. Ultimately, this co-design approach requires hardware-specific DSLs which explicitly represent the native executions on neutral atom hardware: in other words, Bloqade.


## Hybrid computing beyond circuits

Many quantum algorithms are hybrid, requiring both classical and quantum resources to work together in tandem. This could be anything from syndrome extraction and measurement-based computing to variational parameter updates in VQE methods and orbital fragmentation methods in molecular simulation. Through the use of the Kirin compiler infrastructure, Bloqade embraces this philosophy of heterogeneous compute. Kirin programs are written as (compositions of) [kernels](https://en.wikipedia.org/wiki/Compute_kernel) -- subroutines that are intended to run on particular hardware (such as QPUs), or orchestrated to run on heterogeneous compute (such as a real-time classical runtime plus a QPU). These subroutines -- plus the built-in hybrid representations-- enable many key primitives, such as error correction.

Additionally, the ability to compose functions together and to use typical classical programming structures like `if` and recursions enables many simplifications in writing complex circuits. In fact, recursions and the ability to dynamically allocate new memory (which is not known until runtime) enables many powerful subroutines and is natively enabled with Bloqade's kernel-based representation; for example, see [this implementation](../../../digital/examples/repeat_until_success.py) of a repeat-until-success program.

## Analog, digital, logical: towards real quantum utility

The first step in Bloqade's journey was building out the analog mode SDK, designed to interface with QuEra’s cloud-accessible analog-mode neutral-atom quantum computer Aquila, as well as enable analysis and scientific discovery in analog quantum computing. But the journey should not stop there: real quantum utility is error corrected and requires robust algorithmic exploration and design of quantum primitives, in-depth analysis of near-term hardware performance and benchmarking, and building pipelines and hybrid architectures that are intended not just for today’s demonstrations but also for tomorrow’s utility-scale hardware. By introducing the next generation of Bloqade, we hope to enable this exploration by adding in support for near-term digital and intermediate-term logical representations of hybrid quantum computations.

## Learn more

Bloqade is an open-source project and can be freely downloaded and modified; you can learn how to do so [here](../../../install.md). If you want to see how to write programs with of the new `bloqade` package, check out our examples [here](../../../digital/index.md). If you would like to learn more about QuEra Computing Inc., check out our [webpage](https://quera.com) as well as discover our many [academic publications and demonstrations](https://www.quera.com/news#publications).
