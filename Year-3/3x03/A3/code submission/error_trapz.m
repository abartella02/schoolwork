function [c, k] = error_trapz(f, a, b)

    F = integral(f, a, b);
    
    points = 10000;
    
    [h, ln, E, E_ln] = deal(zeros(points, 1));
    
    for i=1:points
        h(i) = (b-a)/(i*10);
        ln(i) = log(h(i));
    end

    for i=1:points
        x = a:h(i):b;
        t = trapz(x, f(x));
        E(i) = abs(F-t);
        E_ln(i) = log(E(i));
    end

    A = [sum(ln.^2) sum(ln); sum(ln) points];
    B = [sum(ln.*E_ln); sum(E_ln)];

    y = A\B;
    k = y(1);
    c = exp(y(2));

end