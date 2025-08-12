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

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Name tellus sem, mattis eu quam in, semper semper turpis. Suspendisse dignissim sagittis dui imperdiet sagittis. Duis imperdiet rutrum turpis eu pulvinar. Phasellus bibendum porta fermentum. Phasellus eu lobortis lectus. Duis massa risus, porttitor id metus quis, sollicitudin dapibus orci. Ut tincidunt ultrices diam, sit amet molestie purus accumsan quis. Ut erat felis, molestie eu orci sed, ultrices sagittis augue.

## Circuit-level compared to hardware-level programming (David)

Gemini class devices are digital quantum computers.
This allows you to work on the circuit-level of abstraction rather than the hardware-level.
While this is certainly useful, in the current era of noisy intermediate scale quantum devices, you inevitable have to consider potential noise processes when developing quantum programs.

When writing a circuit, noise processes can be taken into account as channels that cause decoherence thereby reducing the overall circuit fidelity.
If the fidelity is too low, the computation may contain errors.
Before executing a circuit, you need to know whether this circuit will actually lead to the desired results.
This is where emulation comes in, which, in order to faithfully represent the results you can expect, needs to account for noise.

At the hardware level, you always need to work with (or around) the noise that is adherent to the hardware you are programming on.
This comes at the loss of abstraction and subsequently high-level tooling.
At the same time, however, in enables you to device your own strategies in order to suppress noise in your specific application, which will oftentimes outperform today's compilers.
Here, we will focus on circuit level programming, but please refer to [bloqade-shuttle](https://queracomputing.github.io/bloqade-shuttle/dev/) to learn more about our hardware-level programming capabilities.

Even including noise channels, circuit-level programming remains abstract in that you do not have to consider the specific hardware you are running the circuit on.
However, it is the very nature of the noise channels, where the details of the hardware come into play.
To know whether a circuit will execute with a sufficiently high fidelity, the noise parameters and channels need to represent the infidelity of the gates executed on the particular hardware (in this case Gemini).

In order to provide users with the required set of tools, we have spent considerable time researching and implementing an easy-to-use framework that allows you to include Gemini's particular noise processes in a high-level circuit.

## Heuristic approach to noise (Tyler)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Name tellus sem, mattis eu quam in, semper semper turpis. Suspendisse dignissim sagittis dui imperdiet sagittis. Duis imperdiet rutrum turpis eu pulvinar. Phasellus bibendum porta fermentum. Phasellus eu lobortis lectus. Duis massa risus, porttitor id metus quis, sollicitudin dapibus orci. Ut tincidunt ultrices diam, sit amet molestie purus accumsan quis. Ut erat felis, molestie eu orci sed, ultrices sagittis augue.

## Example: Noise in the GHZ state

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Name tellus sem, mattis eu quam in, semper semper turpis. Suspendisse dignissim sagittis dui imperdiet sagittis. Duis imperdiet rutrum turpis eu pulvinar. Phasellus bibendum porta fermentum. Phasellus eu lobortis lectus. Duis massa risus, porttitor id metus quis, sollicitudin dapibus orci. Ut tincidunt ultrices diam, sit amet molestie purus accumsan quis. Ut erat felis, molestie eu orci sed, ultrices sagittis augue.

### Flow chart (Tyler)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Name tellus sem, mattis eu quam in, semper semper turpis. Suspendisse dignissim sagittis dui imperdiet sagittis. Duis imperdiet rutrum turpis eu pulvinar. Phasellus bibendum porta fermentum. Phasellus eu lobortis lectus. Duis massa risus, porttitor id metus quis, sollicitudin dapibus orci. Ut tincidunt ultrices diam, sit amet molestie purus accumsan quis. Ut erat felis, molestie eu orci sed, ultrices sagittis augue.

### Annotated circuit (Luis)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Name tellus sem, mattis eu quam in, semper semper turpis. Suspendisse dignissim sagittis dui imperdiet sagittis. Duis imperdiet rutrum turpis eu pulvinar. Phasellus bibendum porta fermentum. Phasellus eu lobortis lectus. Duis massa risus, porttitor id metus quis, sollicitudin dapibus orci. Ut tincidunt ultrices diam, sit amet molestie purus accumsan quis. Ut erat felis, molestie eu orci sed, ultrices sagittis augue.

### GHZ data (David)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Name tellus sem, mattis eu quam in, semper semper turpis. Suspendisse dignissim sagittis dui imperdiet sagittis. Duis imperdiet rutrum turpis eu pulvinar. Phasellus bibendum porta fermentum. Phasellus eu lobortis lectus. Duis massa risus, porttitor id metus quis, sollicitudin dapibus orci. Ut tincidunt ultrices diam, sit amet molestie purus accumsan quis. Ut erat felis, molestie eu orci sed, ultrices sagittis augue.
