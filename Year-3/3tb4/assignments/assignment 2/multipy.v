module circuit(input [7:0] a, b, output reg [7:0] c, output reg overflow);
  reg [15:0] c_adj, c_temp;
  always @(*) begin
    c_temp = (a*b);
    c_adj = c_temp >> 2;
    
    if(c_temp > 8'hff) //overflow check (not working)
      overflow = 1'b1; //
    else			   //
      overflow = 1'b0; //
    
    c[7:0] = c_adj[7:0];
  end
    
endmodule