f = @(x,y) [x^2+y^2 - 25; x^2 - y-1];
jacdiffs(f)
function J = jacdiffs(F, x)
    %computes the Jacobian oF a Function F: R^2->R^2
    %using Finite diFFerences

    %evaluate F at x
    Fx = F(x);

    h = 1e-8;
    sigma = x(1);
    %perturb the First component oF x
 
    X = [x(1) + sigma * h; x(2)]; % x(1)*h
    %First column oF Jacobian
    J(:, 1) = (F(X) - Fx) ./ (sigma * h);
    %perturb the second component oF x
    sigma = x(2);
    X = [x(1); x(2) + sigma * h];
    %second column oF Jacobian
    J(:, 2) = (F(X) - Fx) ./ (sigma * h);
end