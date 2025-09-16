//DO NOT USE RN
module buses_2_d (input clk, input [15:0] signal, input [1:0] sel, 
output reg [15:0] result); 
    wire [15:0] bus [3:0];

    assign bus = 
    
    always @(*) begin
        case(select) 
            2'b00: out = sample;
            2'b01: out = sample_fir;
            2'b10: out = sample_echo;
            default: out = sample; //default case
        endcase
    end
endmodule