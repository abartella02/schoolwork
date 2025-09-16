syms d1 d2 d3 a1 a2 a3 theta1 theta2 theta3 q1 q2 q3

T = [0 0 1 0;
    0 sin(theta3) cos(theta3) d2+a3*sin(theta3);
    -cos(theta3) sin(theta3) 0 d1-a3*cos(theta3);
    0 0 0 1]

P_x = T(1,4);
P_y = T(2,4);
P_z = T(3,4);

q1 = d1;
q2 = d2;
q3 = theta3;

J_A = [diff(P_x,q1) diff(P_x,q2) diff(P_x,q3);
    diff(P_y,q1) diff(P_y,q2) diff(P_y,q3);
    diff(P_z,q1) diff(P_z,q2) diff(P_z,q3)]

zeta1 = 0;
zeta2 = 0;
zeta3 = 1;

z0 = [0;0;1];

A1 = [-1 0 0 0;
    0 0 1 0;
    0 1 0 d1;
    0 0 0 1];

A2 = [0 0 -1 0;
    -1 0 0 0;
    0 1 0 d2;
    0 0 0 1];

A3 = [cos(theta3) -sin(theta3) 0 a3*cos(theta3);
    sin(theta3) cos(theta3) 0 a3*sin(theta3);
    0 0 1 0;
    0 0 0 1];

z1 = A1(1:3,1:3)*z0
z2 = A1(1:3,1:3)*A2(1:3,1:3)*z0
%z3 = A1(1:3,1:3)*A2(1:3,1:3)*A3(1:3,1:3)*z0

J_B = [zeta1*z0 zeta2*z1 zeta3*z2]

J = [J_A; J_B]
