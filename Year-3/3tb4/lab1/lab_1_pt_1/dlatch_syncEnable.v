module dlatch_syncEnable(input D, input clk, input enable, output reg Q);
  always @(posedge clk) begin
    if (enable) begin
      Q <= D;
    end
  end
endmodule