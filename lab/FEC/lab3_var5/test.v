`timescale 1ns/1ps

module lab3_tb;

reg [31:0] a, x;
reg clk, rst;
reg start;

wire busy;
wire [63:0] y;

final final_utt(
	.clk_i(clk),
	.rst_i(rst),
	.start_i(start),
	.a_bi(a),
	.b_bi(x),
	.busy_o(busy),
	.y_bo(y)
);

always #5 clk = ~clk;

always @(negedge clk) begin
    a = ($random % 4294967296) & 8'hFFFFFFFF;
    x = ($random % 4294967296) & 8'hFFFFFFFF;
end

initial begin
    $dumpfile("time.vcd");
    $dumpvars(0, lab3_tb);

    clk   = 0;
    rst   = 1;
    x     = 0;
    a     = 0;
    start = 0;

    #20
    rst   = 0;
    start = 1;

    #500
    $finish;
end
    
endmodule