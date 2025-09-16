module four_bit_counter(input enable, 
                        input low_reset, clk,
                       output reg [3:0] out);
  always @(posedge clk) begin
    if(~low_reset)
      out <= 4'b0;
    else if(enable)
      out <= out + 4'b0001;
  end
endmodule