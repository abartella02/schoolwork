function error(tol)
    %trapezoidal
    n_trap = ceil(((exp(pi)*pi^3)/(32*tol))^(1/2));
    E_trap = (exp(pi)*pi^3)/(32*n_trap^2);
    
    %simpson
    n_s = ceil(((7*exp(pi)*pi^5)/(5760*tol))^(1/4));
    if(mod(n_s, 2)~=0)
        n_s = n_s + 1;
    end
    E_s = (7*exp(pi)*pi^5)/(5760*n_s^4);

    %midpoint
    n_mid = ceil(((exp(pi)*pi^3)/(64*tol))^(1/2));
    E_mid = (exp(pi)*pi^3)/(64*n_mid^2);

    fprintf("tol = %d\n", tol)
    fprintf("trapezoid n= %i, error=%d\n", n_trap, E_trap)
    fprintf("midpoint  n= %i, error=%d\n", n_mid, E_mid)
    fprintf("simpson   n= %i, error=%d\n", n_s, E_s)

end