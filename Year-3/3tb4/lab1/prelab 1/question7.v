module seven_seg_decoder(input [3:0] x, output [6:0] hex_LEDs);
    reg [6:0] reg_LEDs;
  
    assign hex_LEDs[0] = (~x0&~x1&x2&~x3) | (x0&x2&x3) | (x0&x1&x3);
    assign hex_LEDs[1] = (~x0&~x1&x2&x3) | (x0&~x1&x2&~x3) | (x0&x1&x3) | (~x0&x1&x2&~x3);

    assign hex_LEDs[6:2] = reg_LEDs[6:2];

    always @(*) begin
        case(x)
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
            4'b1011: reg_LEDs[6:2] = 5'b00000;
            4'b1100: reg_LEDs[6:2] = 5'b10001;
            4'b1101: reg_LEDs[6:2] = 5'b01000;
            4'b1110: reg_LEDs[6:2] = 5'b00001;
            4'b1111: reg_LEDs[6:2] = 5'b00011;
        endcase
    end
endmodule