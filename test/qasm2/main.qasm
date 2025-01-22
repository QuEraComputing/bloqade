OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

h q[0];
cx q[0], q[1];
rz(2.0+1+1, theta, alpha + 1) q[0], q[1];
rx(pi) q[0];
barrier q;
CX q[0], q[1];
U(+sin(pi/2), pi, pi*3) q[1];
MOVE q[0] -> q[1];

gate foo a, b {
  h a;
  cx a, b;
  h a;
}

gate rx(alpha, theta) qreg_1 {
  rz(theta, alpha) qreg_1;
  ry(-1.5707963267948966) qreg_1;
  rz(1.5707963267948966) qreg_1;
}

measure q -> c;
measure q[1] -> c[1];

if (c == 1) x q[0];
if (c == 1) {
    x q[0];
    x q[0];
}
