
a = 0; b = 1;
f = @(x) 100./sqrt(x+0.01) + 1./((x-0.3).^2+0.1) - pi;
[c,k] = error_trapz(f,a,b);
fprintf('c=%e  k=%g\n', c, k)

f = @(x) exp(-x.^2);
[c,k] = error_trapz(f,a,b);
fprintf('c=%e  k=%g\n', c, k)