f = @(x) [x(1) + x(2)*(x2*(5-x(2))-2)-13; x(1) + x(2)*(x(2)*(1+x2)-14)-29];
jacobian(f)
J = jacobian(f);
x0 = [1, 0]';
J(x0)
%matlabFunction(J, 'Vars', {argnames(J).'})
% f = @(x, y)[x^2+y^2 - 25; 
%      x^2 - y-1];
% sym(f)
% 
% diff(f, x)

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