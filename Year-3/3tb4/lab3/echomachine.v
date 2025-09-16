module echomachine(input clk, input [15:0] signal, 
output reg [15:0] signal_out);
    wire [15:0] delayed_signal, delayed_att_signal, echo_signal;
    assign echo_signal = signal_out;
    shift(.clock(clk), .shiftin(echo_signal), .shiftout(delayed_signal));
    
    assign delayed_att_signal = delayed_signal >>> 2;
    
    always @(posedge clk) begin
        signal_out = signal + delayed_att_signal;
    end
endmodule