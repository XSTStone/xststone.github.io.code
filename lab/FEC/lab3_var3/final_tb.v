`timescale 1ns/1ps

module final_tb;

reg [31:0] root_x;
reg root_clk, root_rst, root_start;

wire [2:0] root_state;
wire [31:0] root_y;

reg [31:0] square_a, square_b;
reg square_clk, square_rst, square_start;

wire [2:0] square_state;
wire [63:0] square_y_a, square_y_b;

root root_utt(
    .clk_i(root_clk),
    .rst_i(root_rst),
    .start_i(root_start),

    .x_bi(root_x),

    .state_o(root_state),
    .y_bo(root_y)
);

square square_utt(
    .clk_i(square_clk),
    .rst_i(square_rst),
    .start_i(square_start),

    .a_bi(square_a),
    .b_bi(square_b),

    .y_a_bo(square_y_a),
    .y_b_bo(square_y_b),
    .state_o(square_state)
);

always #5 root_clk = ~root_clk;
always #5 square_clk = ~square_clk;

always @(negedge root_clk) begin
    // root logic
    if (square_state == 0) begin
        square_a = ($random % 256) & 8'hFF;
        square_b = ($random % 256) & 8'hFF;
    end

    if (square_state == 2 && root_state == 0) begin
        root_rst = 0;
        root_start = 1;
        root_x = square_y_a + square_y_b;
    end

    if (square_state == 2 && root_state == 2) begin
        root_rst = 1;
        root_start = 0;
        root_x = 0;

        square_rst = 1;
        square_start = 0;
        square_a = 0;
        square_b = 0;

        #20
        square_rst = 0;
        square_start = 1;
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

    // square reset logic
    square_rst = 1;
    square_clk = 0;
    square_start = 0;
    square_a = 0;
    square_b = 0;

    #20
    // square logic
    square_rst = 0;
    square_start = 1;

    #1200
    $finish;
end

endmodule
