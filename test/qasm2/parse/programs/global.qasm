KIRIN {qasm2.glob, qasm2.uop};
include "qelib1.inc";

qreg q1[2];
qreg q2[3];

glob.U(1.0, 2.0, 3.0) {q1, q2}
