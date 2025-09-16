module thirty_bit_counter(input enable, 
                         input low_reset,
                         output reg [29:0] out);
  always @(posedge enable or negedge low_reset) begin
    if(~low_reset) begin
      out <= 30'b0;
        end
    else if(enable) begin
      out <= out + 30'b1;
        end
  end
  
  
endmodule