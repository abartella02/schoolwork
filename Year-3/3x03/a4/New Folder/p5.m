close all;
format long

odefun = @(t, y) [-0.1*y(1)-199.9*y(2);0*y(1)-200*y(2)];

odefunPrime = @(t, y) [-0.1*y(1)-199.9*y(2);0*y(1)-200*y(2)];

y0 = [2, 1];
tspan = [0, 100];

[t, state] = ode23s(odefun, tspan, y0);
[t2, state2] = ode45(odefun, tspan, y0);

y1 = exp(-0.1*t)+exp(-200*t);
y2 = exp(-200*t);

tolerances = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, ...
    1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12];

x1 = linspace(0, 100);

y1sol = y1(83);

ystate1 = state(:, 1);
ystate2 = state(:, 2);

table = zeros(12, 4);

for i = 1:length(tolerances)
    error = [];
    t1 = [];

    tdiff = 0;
    tprev = 0;

    tdiff23 = 0;
    tprev23 = 0;

    for j = 1:length(t)
        if abs(y1(j)-ystate1(j)) <= tolerances(i)
            error = [error, abs(y1(j)-ystate1(j))];
            tdiff = tdiff + abs(t(j)-tprev);
            t1 = [t1, t(j)];
            tprev = t1(end);
        end
    end

    if i == 7
        stepsize = zeros(1, length(t1)-1);
        stept = zeros(1, length(t1) - 1);
        for k = 2:length(t1)
            stepsize(k) = t1(k) - t1(k-1);
            stept(k) = t1(k);
        end
    end

    table(i, 1) = tolerances(i);
    table(i, 2) = length(t1);
    table(i, 3) = abs(y1(83) - ystate1(end));
    table(i, 4) = tdiff / length(t1);
end

disp("           Tol       |    # of Evals     |   error @ t=83   |    avg step sz");
disp(table)

plot(stept(1:end), stepsize(1:end), '-o','LineWidth', 1)
title("Step Size vs t")
xlabel('t');
ylabel('Step Size');
