`timescale 1ns/1ps

module lab3_tb;

reg [31:0] a, b, x;
reg [15:0] square_x;
reg clk, rst;
reg start;

wire busy;
wire [63:0] y;
wire [7:0] square_y;

mult mult_utt(
    .clk_i(clk),
    .rst_i(rst),
    .a_bi(a),
    .start_i(start),
    .busy_o(busy),
    .y_bo(y)
);

square square_utt(
    .clk_i(clk),
    .rst_i(rst),
    .x_bi(suqare_x),
    .start_i(start),
    .busy_o(busy),
    .y_bo(square_y)
);

root root_utt(
    .clk_i(clk),
    .rst_i(rst),
    .x_bi(x),
    .start_i(start),
    .busy_o(busy),
    .y_bo(y)
);

final final_utt(
    .clk_i(clk),
    .rst_i(rst),
    .start_i(start),
    .a_bi(a),
    .b_bi(b),
    .busy_o(busy),
    .y_bo(y)
);

always #5 clk = ~clk;

always @(negedge clk) begin
    a = ($random % 256) & 8'hFF;
    b = ($random % 256) & 8'hFF;
    x = ($random % 256) & 8'hFF;
    square_x = ($random % 256) & 8'hFF;
end

initial begin
    $dumpfile("time.vcd");
    $dumpvars(0, lab3_tb);

    clk   = 0;
    rst   = 1;
    x     = 0;
    b     = 0;
    a     = 0;
    square_x = 0;
    start = 0;

    #20
    rst   = 0;
    start = 1;

    #500
    $finish;
end
    
endmodule