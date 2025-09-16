module add(input signed [7:0] A, B, output reg signed [8:0] C);
    reg signed [8:0] a, b, c_temp;
    reg signed [9:0] c;
    always @(*) begin
        a[8] = A[7]; //preserve sign bits during left padding
        b[0] = 1'b0; //extend least significant bit (right padding)
        a[7:0] = A[7:0];
        b[8:1] = B[7:0];

        c = a + b; //calculate (result in Q2.7)

        c_temp[8:0] = c[8:0]; //remove most significant bit
        C[8:0] = c_temp[8:0] >>> 1; //shift right
    end
    
endmodule

