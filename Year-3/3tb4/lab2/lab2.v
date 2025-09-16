module lab2(input clk, input [2:0] KEY, output [6:0] HEX0, HEX1, HEX2, HEX3, HEX4, HEX5) 
reg [2:0] state = 3'd0;
reg [3:0] LED_status = 4'b0;
reg reset = 1'b0;

always @(*) begin
    case(state)
        3'd0: begin //state 0
            //HEX LEDs BLINK, LEDRs OFF
            blink blink0(.clk(clk), .state(LED_status));
            sevensegdisp ssd0(.SW(LED_status), HEX0); 
            sevensegdisp ssd1(.SW(LED_status), HEX1);
            sevensegdisp ssd2(.SW(LED_status), HEX2);
            sevensegdisp ssd3(.SW(LED_status), HEX3);
            sevensegdisp ssd4(.SW(LED_status), HEX4);
            sevensegdisp ssd5(.SW(LED_status), HEX5);
            //if 5 seconds passes state = 1
        end
        3'd1: begin //state 1
            reset = 1'b0;
            //HEX LEDs off
            sevensegdisp ssd0(.SW(4'b1111), HEX0);
            sevensegdisp ssd1(.SW(4'b1111), HEX1);
            sevensegdisp ssd2(.SW(4'b1111), HEX2);
            sevensegdisp ssd3(.SW(4'b1111), HEX3);
            sevensegdisp ssd4(.SW(4'b1111), HEX4);
            sevensegdisp ssd5(.SW(4'b1111), HEX5);
            //if KEY0 pressed, state = 2
            if(!KEY[0])
                state = 3'd2;
            //if KEY3 pressed, state = 3
            else if(!KEY[3])
                state = 3'd3;
            else if(!KEY[1]) //reset
                state = 3'd0;
                //reset scores too?
                reset = 1'b1;
        end
        3'd2: begin //state 2
            //display 1's on HEX LEDs
            sevensegdisp ssd0(.SW(4'b1), HEX0);
            sevensegdisp ssd1(.SW(4'b1), HEX1);
            sevensegdisp ssd2(.SW(4'b1), HEX2);
            sevensegdisp ssd3(.SW(4'b1), HEX3);
            sevensegdisp ssd4(.SW(4'b1), HEX4);
            sevensegdisp ssd5(.SW(4'b1), HEX5);
            //if KEY1 pressed, state = 0
            if(!KEY[1])
                state = 3'd0;
                //reset scores too?
                reset = 1'b1;
        end
        3'd3: begin
            //display 3's on HEX LEDS
            sevensegdisp ssd0(.SW(4'b11), HEX0);
            sevensegdisp ssd1(.SW(4'b11), HEX1);
            sevensegdisp ssd2(.SW(4'b11), HEX2);
            sevensegdisp ssd3(.SW(4'b11), HEX3);
            sevensegdisp ssd4(.SW(4'b11), HEX4);
            sevensegdisp ssd5(.SW(4'b11), HEX5);
            //if KEY2 pressed, state = 0
            if(!KEY[1])
                state = 3'd0;
                //reset scores too?
                reset = 1'b1;
            if (!KEY[2])
                state = 3'd0;
        end
        3'd4: begin
            //Timer starts
            //HEX LEDs display timer in ms
            
            //if KEY0 pressed, state = 5
            //if KEY3 pressed, state = 6
            //if KEY1 pressed, state = 0
        end
        3'd5: begin
            //Timer stops
            /*turn on another LEDR corresponding to KEY3
                - might need another module for this*/
            //if KEY2 pressed, state = 0
        end
        3'd6: begin
            //Timer stops
            /*turn on another LEDR corresponding to KEY0
                - might need another module for this*/
            //if KEY2 pressed, state = 0
        end
        default: state = 3'd0;
    endcase
end


endmodule