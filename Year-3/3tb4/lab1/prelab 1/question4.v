module dflipflop(input D, input clk, output reg Q);
  always @(posedge clk) begin
  Q <= D;
  end
endmodule

module dff_lowsynchreset(input D, input clk, input reset, output reg Q);
  always @(posedge clk) begin
    if (reset)
      Q <= 1'b0;
    else
  	  Q <= D;
  end
endmodule

module dlatch_synchenable(input D, input clk, input enable, output reg Q);
  always @(posedge clk) begin
    if (enable) begin
      Q <= D;
    end
  end
endmodule

module dff_lowsynchresetenable(input D, input clk, 
                               input reset, input enable,
                               output reg Q);
  always @(posedge clk) begin
    if (reset) begin
      Q <= 1'b0;
    end
    
    else if (enable) begin
      Q <= 1'b1;
    end
    
    else begin
  	  Q <= D;
    end
    
  end
endmodule


module dflipflop(input D, input clk, output reg Q);
  always @(posedge clk) begin
    Q <= D;
  end
endmodule