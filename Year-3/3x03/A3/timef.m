function t = timef(f,A,B)
    tic; 
    C = f(A, B);  
    t = toc;
    if t > 2 % at least two seconds
        return;
    end
    rep = ceil(2/t);
    [t rep];
    tic;
    for i=1:rep
        C = f(A,C);
    end
    t = toc/rep;
end

