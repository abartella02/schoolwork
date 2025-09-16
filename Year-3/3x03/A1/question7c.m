x = pi/4;
h = 10.^(-16:.1:.1);
f = @(x) sin(x).*exp(cos(x));
fpaccurate = (cos(x)-(sin(x)).^2).*exp(cos(x));
fp = (f(x+h)-f(x-h))./(2*h);
error = abs(fpaccurate - fp);
M = 1;
loglog(h, error,'.', 'MarkerSize', 10);
hold on;
loglog(h, (1/6)*M*h.^2+eps./h, 'LineWidth',2);
hold on

h_calculated = (3*eps/M)^(1/3)
xline(h_calculated, 'LineWidth',2)
%loglog([h_calculated h_calculated], [10^(-13) 10], 'Linewidth', 2)
%loglog(h_calculated, 0, 'Linewidth', 2);
hold off

xlabel('h'); ylabel('Error');
title("Approximating f'(x) = (f(x+h)-f(x-h))/(2h) at x=pi/4");
xlim([h(1) h(end)]);
legend('Error in (f(x+h)-f(x-h))./h', '(1/6)*M*h^2+eps./h', 'h=(3*eps/M)^1^/^3')
set(gca, 'FontSize', 12);
print("-depsc2", "deriverr.eps")