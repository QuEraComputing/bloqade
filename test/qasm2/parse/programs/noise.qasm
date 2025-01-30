KIRIN {qasm2.uop, qasm2.noise};
include "qelib1.inc";

qreg q[2];
noise.PAULI1(1.0, 2.0, 3.0) q[0];
