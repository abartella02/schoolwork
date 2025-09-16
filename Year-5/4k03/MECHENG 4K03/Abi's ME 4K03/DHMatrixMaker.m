
% syms theta1 theta2 theta3 theta alpha alpha1 alpha2 alpha3 d d1 d2 d3 a a1 a2 a3
% 
% theta = theta1;
% d = 0;
% a = a1;
% alpha = 0;
% 
% A = [cosd(theta) -sind(theta)*cosd(alpha) sind(theta)*sind(alpha) a*cosd(theta);
%     sind(theta) cosd(theta)*cosd(alpha) -cosd(theta)*sind(alpha) a*sind(theta);
%     0 sind(alpha) cosd(alpha) d;
%     0 0 0 1]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


syms x theta_1 theta_2 theta_3 d_1 d_2 d_3 a_1 a_2 a_3 alpha_1 alpha_2 alpha_3
format("short")

theta_1;
theta_2;
theta_3;

d_1 = 0;
d_2 = 0;
d_3 = 0; 

a_1;
a_2;
a_3;

alpha_1 = 0;
alpha_2 = 0;
alpha_3 = 0;

theta = [theta_1 theta_2 theta_3];
d = [d_1 d_2 d_3];
a = [a_1 a_2 a_3];
alpha = [alpha_1 alpha_2 alpha_3];


rot_Z = @(x)[cosd(theta(x)) -sind(theta(x)) 0 0;sind(theta(x)) cosd(theta(x)) 0 0; 0 0 1 0; 0 0 0 1];
trans_d = @(x)[1 0 0 0;0 1 0 0; 0 0 1 d(x); 0 0 0 1];
trans_a = @(x)[1 0 0 a(x);0 1 0 0 ; 0 0 1 0; 0 0 0 1];
rot_X = @(x)[1 0 0 0; 0 cosd(alpha(x)) -sind(alpha(x)) 0; 0 sind(alpha(x)) cosd(alpha(x)) 0; 0 0 0 1];

A = @(x) rot_Z(x)*trans_d(x)*trans_a(x)*rot_X(x)

% x = n+1 value

A_1 = A(1)
A_2 = A(2)
A_3 = A(3)

T = simplify(A_1*A_2*A_3)

sympref('AbbreviateOutput',false);