OPENQASM 2.0-atom;
include "qelib1.inc";

qreg q[2];
noise.PAULI1(1.0, 2.0, 3.0) q[0];
