clear all; close all;
xpts = linspace(1.94, 2.07, 100);
f = @(x) (x-2)^9;
f_exp = @(x) x^9-18*x^8+144*x^7-672*x^6+2016*x^5-4032*x^4+5376*x^3-4608*x^2+2304*x-512;
c = [1, -18, 144, -672, 2016, -4032, 5376, -4608, 2304, -512];

%a
y_horner = zeros(1, length(xpts));
y_real = zeros(1, length(xpts));
f_horner = hornerfunc(c);

for i=1:length(xpts)
    y_horner(i) = f_horner(xpts(i));
    y_real(i) = f(xpts(i));
end

figure(1)
plot(xpts, y_real, 'om')
title("Actual")

figure(2)
plot(xpts, y_horner, 'ob')
title("Horner")

%b
bisection(f_horner, 1.94, 2.07, exp(-6))


function h = hornerfunc(c)
    syms x
    counter = c(1);
    for i=2:length(c)
        counter = counter*x + c(i);
    end
    h = matlabFunction(counter);
    
end


function [r, iter] = bisection(f, a, b, tol)
    fa = f (a);
    fb = f (b);
    if sign(fa) == sign(fb) 
            r = NaN;
        return
    end
    error = b - a;
    nmax = ceil(log((b-a)/(2*tol))/log(2));
    for n = 0 : nmax 
        error = error/2;
        c = a + error;
        fc = f (c);
        if error < tol
            r = c;
            iter=n;
            break;
        end
        if sign(fa) ~= sign(fc) 
            b = c;
            fb = fc;
        else
            a = c; fa = fc;
        end     
    end
end