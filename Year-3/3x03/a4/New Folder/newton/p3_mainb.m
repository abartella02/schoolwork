
fa1 = @(x1, x2) [x1+x2*(x2*(5-x2)-2)-13; x1+x2*(x2*(1+x2)-14)-29];
fa2 = @(x) [x(1,1)+x(2,1)*(x(2,1)*(5-x(2,1))-2)-13; x(1,1)+x(2,1)*(x(2,1)*(1+x(2,1))-14)-29];

x0 = [15, -2]';
tol = 10^(-6);

[newta, countera] = newton(fa2, jacobian(fa1), x0, tol);
mat = fsolve(fa2, x0);

fprintf("\n*** Question A ***\n")
fprintf("Newton's method implementation: x1 = %d, x2 = %d" + ...
    "\nNumber of iterations: %i" + ...
    "\nFsolve implementation: x1 = %d, x2 = %d\n", ...
    newta(1,1), newta(2,1), ...
    countera, ...
    mat(1,1), mat(2,1))
same = round(newta, 5) == round(mat, 5);
fprintf("Sameish results? %i\n", same)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fb1 = @(x1, x2, x3) [x1^2+x2^2+x3^2-5; x1+x2-1;x1+x3-3];
fb2 = @(x) [x(1,1)^2+x(2,1)^2+x(3,1)^2-5; x(1,1)+x(2,1)-1;x(1,1)+x(3,1)-3];

x0 = [(1+sqrt(3))/2, (1-sqrt(3))/2, sqrt(3)]';
tol = 10^(-6);

[newtb, counterb] = newton(fb2, jacobian(fb1), x0, tol);
mat = fsolve(fb2, x0);

fprintf("\n*** Question B ***\n")
fprintf("Newton's method implementation: x1 = %d, x2 = %d, x3 = %d" + ...
    "\nNumber of iterations: %i" + ...
    "\nFsolve implementation: x1 = %d, x2 = %d, x3 = %d\n", ...
    newtb(1,1), newtb(2,1), newtb(3,1), ...
    counterb, ...
    mat(1,1), mat(2,1), mat(3,1))
same = round(newtb, 5) == round(mat, 5);
fprintf("Sameish results? %i\n", same)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fc1 = @(x1, x2, x3, x4) [x1+10*x2; sqrt(5)*(x3-x4); (x2-x3)^2; sqrt(10)*(x1-x4)^2];
fc2 = @(x) [x(1, 1)+10*x(2, 1); sqrt(5)*(x(3, 1)-x(4,1)); (x(2, 1)-x(3, 1))^2; sqrt(10)*(x(1, 1)-x(4, 1))^2];

x0 = [1, 2, 1, 1]';
tol = 10^(-6);

[newtc, counterc] = newton(fc2, jacobian(fc1), x0, tol);
mat = fsolve(fc2, x0);

fprintf("\n*** Question C ***\n")
fprintf("Newton's method implementation: x1 = %d, x2 = %d, x3 = %d, x4 = %d" + ...
    "\nNumber of iterations: %i" + ...
    "\nFsolve implementation: x1 = %d, x2 = %d, x3 = %d, x4 = %d\n", ...
    newtc(1,1), newtc(2,1), newtc(3,1), newtc(4,1),...
    counterc, ...
    mat(1,1), mat(2,1), mat(3,1), mat(4,1))
same = round(newtc, 5) == round(mat, 5);
fprintf("Sameish results? %i\n", same)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fd1 = @(x1, x2) [x1; 10*x1/(x1+0.1)+2*x2^2];
fd2 = @(x) [x(1,1); 10*x(1,1)/(x(1,1)+0.1)+2*x(2,1)^2];

x0 = [1.8, 0]';
tol = 10^(-6);

[newtd, counterd] = newton(fd2, jacobian(fd1), x0, tol);
mat = fsolve(fd2, x0);

fprintf("\n*** Question D ***\n")
fprintf("Newton's method implementation: x1 = %d, x2 = %d" + ...
    "\nNumber of iterations: %i" + ...
    "\nFsolve implementation: x1 = %d, x2 = %d\n", ...
    newtd(1,1), newtd(2,1), ...
    counterd, ...
    mat(1,1), mat(2,1))
same = round(newtd, 5) == round(mat, 5);
fprintf("Sameish results? %i\n", same)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [roots, counter] = newton(F,J,x0,tol)
    counter = 0;
    x_n = x0;
    x_nplus1 = x0 - J(x0)\( F(x_n) );
    while norm(x_nplus1 - x_n)>tol
        counter = counter + 1;
        x_n = x_nplus1;
        x_nplus1 = x_n - J(x_n)\F(x_n);
    end
    roots = x_nplus1;
end

function J = jacobian(f)
    f = sym(f);
    vars = symvar(f);

    rows = length(f);
    cols = length(vars);

    j = sym(zeros(rows, cols));
    for r=1:rows
        for c=1:cols
            j(r,c) = diff(f(r,1), vars(c));
        end
    end
    %fprintf("Jacobian: %s\n", char(j))
    J = matlabFunction(j, 'Vars', {symvar(f).'});
end