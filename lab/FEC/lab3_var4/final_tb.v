`timescale 1ns/1ps

module final_tb;

reg [31:0] root_x;
reg root_clk, root_rst, root_start;

wire [2:0] root_state;
wire [31:0] root_y;

reg [31:0] cubic_a, cubic_b;
reg cubic_clk, cubic_rst, cubic_start;

wire busy;
wire [2:0] cubic_state;
wire [63:0] cubic_y_b;

root root_utt(
    .clk_i(root_clk),
    .rst_i(root_rst),
    .start_i(root_start),
    .x_bi(root_x),
    .state_bo(root_state),
    .y_bo(root_y)
);

cubic cubic_utt(
    .clk_i(cubic_clk),
    .rst_i(cubic_rst),
    .x_bi(cubic_b),
    .start_i(cubic_start),
    .busy_o(busy),
    .y_bo(cubic_y_b),
    .state_bo(cubic_state)
);

always #5 root_clk = ~root_clk;
always #5 cubic_clk = ~cubic_clk;

always @(negedge root_clk) begin
    // root logic
    if (cubic_state == 0) begin
        cubic_a = ($random % 256) & 8'hFF;
        cubic_b = ($random % 256) & 8'hFF;
    end

    if (cubic_state == 3 && root_state == 0) begin
        root_rst = 0;
        root_start = 1;
        root_x = cubic_a + cubic_y_b;
    end

    if (cubic_state == 3 && root_state == 3) begin
        root_rst = 1;
        root_start = 0;
        root_x = 0;

        cubic_rst = 1;
        cubic_start = 0;
        cubic_b = 0;

        #20
        cubic_rst = 0;
        cubic_start = 1;
    end
end

initial begin

    $dumpfile("time.vcd");
    $dumpvars(0, final_tb);

    // root reset logic
    root_rst = 1;
    root_clk = 0;
    root_start = 0;
    root_x = 0;

    // cubic reset logic
    cubic_rst = 1;
    cubic_clk = 0;
    cubic_start = 0;
    cubic_b = 0;

    #20
    // cubic logic
    cubic_rst = 0;
    cubic_start = 1;

    #1200
    $finish;
end

endmodule