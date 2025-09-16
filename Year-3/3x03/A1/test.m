for i=0:30
    factorial(i) == myfact(i)
end


function m = myfact(i)
    m = 1;
    if i ~= 0
        for n = 1:i
            m = m*n;
        end
    end
end
