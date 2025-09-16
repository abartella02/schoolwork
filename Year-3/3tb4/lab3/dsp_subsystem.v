module dsp_subsystem(input clk, input [15:0] signal, input [1:0] select, output [15:0] signal_output);
    wire [15:0] signal_fir, signal_echo;
    echomachine(.clk(clk), .signal(signal), .signal_out(signal_echo));
    //fir filter instance
    mux(.clk(clk), .sample(signal), .sample_fir(signal_fir), .sample_echo(signal_echo), 
        .sel1(select[1]), .sel0(select[0]), .out(signal_output));
endmodule