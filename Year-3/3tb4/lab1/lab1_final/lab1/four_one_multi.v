module four_one_multi(input a, b, c, d,
                     input sel0, sel1,
                     output reg out);
  always @(*) begin
    if(~sel0 & ~sel1) //sel = 00
      out = a;
    else if(sel0 & ~sel1) //sel = 01
      out = b;
    else if(~sel0 & sel1) //sel = 10
      out = c;
    else if(sel0 & sel1) //sel = 11
      out = d;
  end
endmodule
