module sevensegdisp (input [3:0] SW, output [6:0] HEX0);
    reg [6:0] reg_LEDs;
 
    assign HEX0[0] = (~SW[3] & SW[2] & ~SW[1] & ~SW[0]) | (~SW[3] & ~SW[2] & ~SW[1] & SW[0]) | 
                    (SW[3] & SW[2] & SW[0]) | (SW[1] & SW[0] & SW[3]);
                                
    assign HEX0[1] = (~SW[0] & ~SW[1] & SW[2] & SW[3]) | (SW[0] & ~SW[1] & SW[2] & ~SW[3]) | 
                                (SW[0] & SW[1] & SW[3]) | (~SW[0] & SW[1] & SW[2] & ~SW[3]);

    assign HEX0[6:2] = reg_LEDs[6:2];

    always @(*) begin
        case(SW)
            //                         b23456
            4'b0000: reg_LEDs[6:2] = 5'b10000;
            4'b0001: reg_LEDs[6:2] = 5'b11110;
            4'b0010: reg_LEDs[6:2] = 5'b01001;
            4'b0011: reg_LEDs[6:2] = 5'b01100;
            4'b0100: reg_LEDs[6:2] = 5'b00110;
            4'b0101: reg_LEDs[6:2] = 5'b00100;
            4'b0110: reg_LEDs[6:2] = 5'b00000;
            4'b0111: reg_LEDs[6:2] = 5'b11110;
            4'b1000: reg_LEDs[6:2] = 5'b00000;
            4'b1001: reg_LEDs[6:2] = 5'b00100;
            4'b1010: reg_LEDs[6:2] = 5'b00010;
            4'b1011: reg_LEDs[6:2] = 5'b10001;
            4'b1100: reg_LEDs[6:2] = 5'b00001;
            4'b1101: reg_LEDs[6:2] = 5'b00010;
            4'b1110: reg_LEDs[6:2] = 5'b00000;
            4'b1111: reg_LEDs[6:2] = 5'b11111;
        endcase
    end
endmodule