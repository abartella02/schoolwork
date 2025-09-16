close all; clear all;

%Alex Bartella, 400308868, Assignment 1

global n;
global a;
global b;
n = 10;
a = 1;
b = 1+2*eps;

for j = 0:n
    fprintf("****i=%i****\n", j)
    fprintf("x1: %f\n  ", x1(j))
    x1(j) == 1
    fprintf("x2: %f\n  ", x2(j))
    x2(j) == 1
end

function x = x1(i)
    global n;
    global a;
    global b;
    h = (b-a)/n;

    if i == 0
        x = a;
    else
        x = x1(i-1)+h;
    end
end

function x = x2(i)
    global n;
    global a;
    global b;
    h = (b-a)/n;

    x = a + i*h;
end
