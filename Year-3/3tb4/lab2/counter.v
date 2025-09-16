module counter(input clk, reset_n,
                input start_n, stop_n,
                output reg [19:0] ms_count);
    reg paused = 1'b0;
    
    always @(negedge stop_n or negedge start_n) begin
        if(!start_n) begin
            paused <= 1'b1;
        end
        else if (!stop_n) begin
            paused <= 1'b0;
        end
    end
                     
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            ms_count <= 20'b0;
        end 
        else if(paused) begin
            ms_count <= ms_count + 20'b1;
        end
        
    end

endmodule