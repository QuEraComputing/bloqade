KIRIN {qasm2.noise, qasm2.parallel, qasm2.uop};
include "qelib1.inc";
qreg q[2];

parallel.U(theta, phi, lam) {q[0]; q[1]; q[2];}
parallel.CZ {
  q[0], q[1];
  q[2], q[3];
}
parallel.RZ(theta) {
  q[0], q[1];
  q[2], q[3];
}
