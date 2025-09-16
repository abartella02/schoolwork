function B = inverse(A)    
[L, U, P]=lu(A);

len = length(A);
B = zeros(size(A));
I = eye(size(A));
    
    for i=1:len
        b = I(:,i);
        B(:,i) = bSub(U, fSub(L, P*b));
    end

end

function x = bSub(U, b)
    len = length(b);
    x = zeros(len, 1);
    x(len) = b(len)/U(len, len);
    for i=(length(b)-1):-1:1
        sum = 0;
        for j=len:-1:i
            sum = sum + U(i, j)*x(j);
        end
        x(i) = (b(i)-sum)/U(i, i);
    end
end

function x = fSub(L, b)
    x = zeros(length(b), 1);
    for i=1:length(b)
        sum = 0;
        for j=i-1:-1:1
            sum = sum + L(i, j)*x(j);
        end
        x(i)=(b(i)-sum)/L(i,i);
    end
end