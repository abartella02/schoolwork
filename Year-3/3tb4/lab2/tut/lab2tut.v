module lab2tut (input CLOCK_50, input [2:0] KEY, output [6:0] HEX0, HEX1, HEX2, HEX3, HEX4, HEX5);
    wire clk_en; 
    wire [19:0] ms; 
    wire [3:0] digit0, digit1, digit2, digit3, digit4, digit5;

    clock_divider #(.factor(50000)) ck_div(.clk(CLOCK_50), .Reset_n(KEY[0]), .clk_ms(clk_en)); 

    counter counter0(.clk(clk_en), .reset_n(KEY[0]), .start_n(KEY[1]), .stop_n(KEY[2]), .ms_count(ms)); 

    hex_to_bcd_converter converter0(.clk(CLOCK_50), .reset(KEY[0]), .hex_number(ms), .bcd_digit_0(digit0),
                                    .bcd_digit_1(digit1), .bcd_digit_2(digit2), .bcd_digit_3(digit3),
                                    .bcd_digit_4(digit4), .bcd_digit_5(digit5));

    sevensegdisp decoder0(digit0, HEX0); 
    sevensegdisp decoder1(digit1, HEX1);
    sevensegdisp decoder2(digit2, HEX2);
    sevensegdisp decoder3(digit3, HEX3);
    sevensegdisp decoder4(digit4, HEX4);
    sevensegdisp decoder5(digit5, HEX5);      

endmodule