module random(input clk, reset_n, resume_n, output reg [13:0] random, output reg rnd_ready)
//for 14 bits liner feedback shift register
//the taps that need ot be XNORed are 14, 5, 2, 1

wire xnor_taps, and_allbits, feedback;

reg [13:0] reg_values;

reg enable=1;

always @(posedge clk, negedge reset_n, negedge resume_n) begin
    if(!reset_n) begin
        reg_values <= 14'b11111111111111;
        //the LFSR cannot be all 0 at beginning
        enable <= 1;
        rnd_ready<=0;
    end else if (!resume_n) begin
        enable <= 1;
        rnd_ready <= 0;
        reg_values <= reg_values;
    end else begin
        if(enable) begin
            reg_values[13] = reg_values[0];
            reg_values[12:5] = reg_values[13:6];
            reg_values[4] <= reg_values[0] ^ reg_values[5];
            //tap 5 of the diagram from the lab manual

            reg_values[3] = reg_values[4];

            reg_values[2] <= reg_values[0] ^ reg_values[3];
            //tap 3 of the diagram from the lab manual

            reg_values[1]=reg_values[2];

            reg_values[0]<=reg_values[0] ^ reg_values[1];
            //tap 1 of the diagram from the lab manual
            
            if(reg_values <= 13'b1001110001000 and reg_values >= 13'b1111101000) begin
                // the number is within the appropriate range
                random <= reg_values;
            end else begin
                // the number is not within the appropriate range
                random <= 13'b1111101000; //1000
            end
        end
    end
end
endmodule