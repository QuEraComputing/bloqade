OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];

parallel_U(theta, phi, lam) {q[0]; q[1]; q[2];}
parallel_CZ {
  q[0], q[1];
  q[2], q[3];
}
