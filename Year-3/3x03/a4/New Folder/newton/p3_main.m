
f1 = @(x1, x2, x3, x4) [x1 + 10*x2; sqrt(5)*(x3-x4); (x2-x3)^2; sqrt(10)*(x1-x4)^2];
f2 = @(x) [x(1,1) + 10*x(2,1); sqrt(5)*(x(3,1)-x(4,1)); (x(2,1)-x(3,1))^2; sqrt(10)*(x(1,1)-x(4,1))^2];

x0 = [1, 2, 1, 1]';
tol = 10^(-6);

newt = newton(f2, jacobian(f1), x0, tol);
mat = fsolve(f2, x0);

fprintf("\nNewton's method implementation: x1 = %d, x2 = %d, x3 = %d, x4 = %d " + ...
    "\nFsolve implementation: x1 = %d, x2 = %d, x3 = %d, x4 = %d\n", ...
    newt(1,1), newt(2,1), newt(3,1), newt(4,1), ...
    mat(1,1), mat(2,1), mat(3,1), mat(4,1))
same = round(newt, 15) == round(mat, 15);
fprintf("Same results? %i\n", same)

function roots = newton(F,J,x0,tol)
    x_n = x0;
    x_nplus1 = x0 - J(x0)\( F(x_n) );
    while norm(x_nplus1 - x_n)>tol
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
    fprintf("Jacobian: %s\n", char(j))
    J = matlabFunction(j, 'Vars', {symvar(f).'});
end