# Bloqade Manifesto

The vision of Bloqade is to empower quantum scientists, from applications development to algorithmic co-design, to build hybrid quantum-classical programs that leverage the strength of neutral atom quantum computers and have a real chance of demonstrating quantum utility. Bloqade is built on top of [Kirin](https://github.com/QuEraComputing/kirin/), an open source compiler infrastructure designed for kernel functions and composable representations.

## Composable quantum programming

Today Bloqade becomes a [namespace package](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/) of multiple eDSLs (embedded domain-specific languages) around digital and analog quantum computation. `bloqade.analog` is the module for analog-mode neutral atom computers and includes several handy utilities ranging from building or analyzing analog programs, to emulation or executing on QuEra's cloud-accessible hardware "Aquila".

Other submodules such as `bloqade.qasm2`, `bloqade.pyqrack` and `bloqade.stim` are the initial iteration to represent digital circuit execution using gate-based quantum computing on reconfigurable neutral atoms. It extends the QASM2 language to include extra annotation of circuits that is important for efficient execution, such as parallelism and global gates. As well as being able to construct quantum programs with the full convenience of typical classical programming within hardware kernels -- such as loops and control flow -- Bloqade also includes basic compiler transformation passes, emulation, and code generation.

But Bloqade is not done with just these modules. We envision adding new modules (called "dialects") which help you write programs which are tuned for optimal performance in an error corrected era, and on neutral atom hardware. Stay tuned and help us build the future of quantum computing as we build out new components, such as QEC and atom moving dialects.


## Hardware-oriented programming and co-design

At its core, Bloqade strives to be the neutral atom SDK for getting the most out of today's and tomorrows' quantum hardware. It is clear that the circuit-level abstraction is not enough to program real quantum hardware; indeed, tomorrows' quantum demonstrations and applications must program at the hardware level and develop special tooling to compile higher-level abstractions to efficient implementations. We call this process **"co-design"**: designing algorithms specialized to near-term hardware, with an eye on nontrivial demonstrations and scalable solutions. Ultimately, this co-design approach requires hardware-specific DSLs which explicitly represent the native executions on neutral atom hardware: in other words, Bloqade.


## Hybrid computing beyond circuits
Many quantum algorithms are hybrid, requiring both classical and quantum resources to work together in a hybrid computation architecture. This could be anything from syndrome extraction and measurement-based computing to variational parameter updates in VQE methods and orbital fragmentation methods in molecular simulation. Through the use of the Kirin compiler infrastructure, Bloqade embraces this philosophy of heterogeneous compute. Kirin programs are written as (compositions of) [kernels](https://en.wikipedia.org/wiki/Compute_kernel)-- subroutines that are intended to run on particular hardware (such as QPUs), or orchestrated to run on heterogeneous compute (such as a real-time classical runtime plus a QPU). These subroutines-- plus the built-in hybrid representations-- enable many key primitives, such as error correction.

Additionally, the ability to compose functions together and to use typical classical programming structures like `if` and recursion enables many simplifications in writing raw circuit executions. In fact, recursion and the ability to dynamically allocate new memory (which is not known until runtime) enables many powerful subroutines and is natively enabled with Bloqade's kernel-based representation; for example, see [this implementation](digital/examples/repeat_until_success.py) of a repeat-until-success program.

## Analog, digital, logical: towards real quantum utility

The first step in Bloqade was building out the analog mode SDK, designed to interface with QuEra’s cloud-accessible analog-mode neutral-atom quantum computer Aquila, as well as enable analysis and scientific discovery in analog quantum computing. But the journey should not stop there: real quantum utility is error corrected and requires robust algorithmic exploration and design of quantum primitives, in-depth analysis of near-term hardware performance and benchmarking, and building pipelines and hybrid architectures that are intended not just for today’s demonstrators but also for tomorrow’s utility-scale hardware. By introducing the next generation of Bloqade, we hope to enable this exploration by adding in support for near-term digital and intermediate-term logical representations of hybrid quantum computations.

## Join us!

If you are interested in contributing, please see the contribution page [here](contrib.md). If you are interested in exploring more about neutral atom quantum computing, check out some analog tutorials [here](https://queracomputing.github.io/bloqade-analog-examples/dev/), and some circuit tutorials [here](https://bloqade.quera.com/latest/digital/). If you wish to work closer with QuEra, please feel free to reach out!
