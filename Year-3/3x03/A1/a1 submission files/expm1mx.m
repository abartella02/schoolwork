close all; clear all;

%Alex Bartella, 400308868, Assignment 1

syms x

f = @(x) (exp(x)-1-x)./x.^2;
N = [-16:1:0];
x = 10.^N;
% compute accurate values in higher precision
accurate = f(vpa(x));
% relative error in f(x)
error_f = abs((f(x)-accurate)./accurate);
loglog(x,error_f, 'o--');
y = expm1mx1(x);
% relative error in expm1mx
error = abs((accurate-y)./accurate);
hold on;
loglog(x,error, 'o--');
legend("rel. error in f(x)", "rel. error in expm1mx")
print("-depsc2", "expm1mx.eps")


function y = expm1mx1(x)
    i = 2; y = 0;
    while i<=18
        %factorial calculator
        fact = 1;
        if i ~= 0
            for n = 1:i
                fact = fact*n;
            end
        end
        %function
        y = y + (x.^(i-2))./fact;
        i = i + 1;
    end
end