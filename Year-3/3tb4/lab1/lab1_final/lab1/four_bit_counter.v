module four_bit_counter(input enable, 
                        input low_reset,
                       output reg [3:0] out);
  always @(posedge enable or negedge low_reset) begin
    if(~low_reset) begin
      out <= 4'b0;
    end
    else if (enable) begin
      out <= out + 4'b0001;
        end
  end
  

endmodule