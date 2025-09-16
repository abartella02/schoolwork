module lab1part3 (input [9:0] SW, input [3:0] KEY, input CLOCK_50, output HEX0);

	reg [3:0] key_press;
	reg [29:0] clock_counter;

    wire [3:0] SW_1, SW_2;
    wire HEX0_1, HEX0_2;

    assign SW_1 = SW;
    assign SW_2 = SW;
	
	
	always @(posedge CLOCK_50) begin
		clock_counter <= clock_counter + 30'b1;
	end
	
	always @(negedge KEY[0]) begin
		clock_counter <= 30'b0;
		key_press <= 4'b0;
	end
		
	always @(negedge KEY[3]) begin
		key_press <= key_press + 4'b1;
	end
	
	always @(*) begin
		case(SW[9:8])
			2'b00: sevensegdisp(.SW(SW),.HEX0(HEX0));
			2'b01: sevensegdisp(.SW(key_press),.HEX0(HEX0));//disp key_press on seven seg disp
			2'b10: sevensegdisp(.SW(clock_counter),.HEX0(HEX0));//disp clock_counter[29:26] on 7sd
		endcase
	end

endmodule
		
		
		
	
		