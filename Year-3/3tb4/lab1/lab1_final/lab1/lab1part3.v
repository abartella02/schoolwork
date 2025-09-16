module lab1part3 (input [9:0] SW, input [3:0] KEY, input CLOCK_50, output [6:0] HEX0);

    wire [3:0] key_press;
    wire [29:0] clock_counter;

    wire [3:0] SW_in;
        
    four_bit_counter pressctr(.enable(KEY[3]), .low_reset(KEY[0]), .out(key_press));
    
    thirty_bit_counter thirtyctr(.enable(CLOCK_50), .low_reset(KEY[0]), .out(clock_counter));

	four_one_multi fourmux(.a(SW[3:0]), .b(key_press), .c(clock_counter[29:26]), .d(4'b1111),
									.sel0(SW[9]), .sel1(SW[8]), .out(SW_in));

    sevensegdisp ssd(.SW(SW_in),.HEX0(HEX0));

endmodule