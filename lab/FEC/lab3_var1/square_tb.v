`timescale 1ns/1ps

module lab3_tb;

reg [15:0] x;
reg clk, rst;
reg start;

wire busy;
wire [63:0] y;
wire [7:0] square_y;

square square_utt(
    .clk_i(clk),
    .rst_i(rst),
    .x_bi(x),
    .start_i(start),
    .busy_o(busy),
    .y_bo(square_y)
);

always #5 clk = ~clk;

always @(negedge clk) begin
    x = 256;
end

initial begin
    $dumpfile("square_time.vcd");
    $dumpvars(0, lab3_tb);

    clk   = 0;
    rst   = 1;
    x = 0;
    start = 0;

    #20
    rst   = 0;
    start = 1;

    #500
    $finish;
end
    
endmodule