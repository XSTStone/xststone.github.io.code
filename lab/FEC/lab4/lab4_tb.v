`timescale 1ns/1ps
module lab4_tb;


reg clk, rst;
wire [31:0] HRDATA;
wire [31:0] HADDR;
wire [31:0] HWDATA;
wire HWRITE;

master muut(
    .HCLK_i(clk),
    .HRESETn_i(rst),
    .HRDATA_bi(HRDATA),
    .HADDR_bo(HADDR),
    .HWDATA_bo(HWDATA),
    .HWRITE_o(HWRITE)
);

slave suut(
    .HCLK_i(clk),
    .HRESETn_i(rst),
    .HRDATA_bo(HRDATA),
    .HADDR_bi(HADDR),
    .HWDATA_bi(HWDATA),
    .HWRITE_i(HWRITE)
);

always #5 clk = ~clk;


initial begin
    $dumpfile("time.vcd");
    $dumpvars(0, lab4_tb);
    clk = 0;
    rst = 0;

    #20
    rst = 1;

    #100
    $finish;
end 

endmodule
