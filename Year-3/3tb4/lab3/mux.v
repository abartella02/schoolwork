module mux(input clk, input [15:0] sample, sample_fir, sample_echo, input sel1, sel0
           output reg [15:0] out);
    wire [1:0] select;
    assign select = {sel1, sel0};
    always @(*) begin
        case(select) 
            2'b00: out = sample;
            2'b01: out = sample_fir;
            2'b10: out = sample_echo;
            default: out = sample; //default case
        endcase
    end
endmodule