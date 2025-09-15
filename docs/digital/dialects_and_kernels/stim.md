# Stim

[Stim](https://github.com/quantumlib/Stim) is a library for simulating and analyzing quantum stabilizer codes and quantum error correction circuits.
The corresponding dialect allows you to write kernels that can then be emitted as native Stim code.

Here is a short example:

```python
from bloqade import stim
from bloqade.stim.emit import EmitStimMain

@stim.main
def main():
    stim.x(targets=(0,2))
    stim.cx(controls=(0,3), targets=(1,3))

main.print()

target = EmitStimMain()
target.run(main, args=())
stim_program = target.get_output()
print(stim_program)
```


See also the [stim API reference](../../reference/bloqade-circuit/src/bloqade/stim/)
