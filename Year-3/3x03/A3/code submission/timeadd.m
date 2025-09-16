function [k_row, c_row, k_col, c_col] = timeadder()
    pts = 1000:500:5000;
    ln = log(pts);
    points = length(pts);
    
    [tR, tC] = deal(zeros(points, 1));
    
    for i=1:points
        a = rand(pts(i));
        b = rand(pts(i));
        tR(i) = timef(@addR, a, b);
        tC(i) = timef(@addC, a, b);
    end

    A1 = [sum(ln.^2) sum(ln); sum(ln) points];
    b1 = [sum(ln*log(tR)); sum(log(tR))];

    A2 = [sum(ln.^2) sum(ln); sum(ln) points];
    b2 = [sum(ln*log(tC)); sum(log(tC))];

    y1 = A1\b1;
    y2 = A2\b2;

    k_row = y1(1);
    c_row = exp(y1(2));

    k_col = y2(1);
    c_col = exp(y2(2));

    fprintf("k_row= %d  c_row= %d\n", k_row, c_row)
    fprintf("k_col= %d  c_col= %d\n", k_col, c_col)
end



function C = addR(A, B)
    [n, ~] = size(A);
    for i = 1:n
        C(i, 1:n) = A(i, 1:n) + B(i, 1:n);
    end
end

function C = addC(A, B)
    [n, ~] = size(A);
    for j = 1:n
        C(1:n, j) = A(1:n, j) + B(1:n, j);
    end
end
