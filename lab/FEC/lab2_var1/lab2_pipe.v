module lab2_pipe (
    input clk,
    input rst,
    input start,
    input [31:0] x,
    output reg rdy,
    output reg [31:0] y
);

reg [31:0] pow2, pow3, pow4, pow5, temp_pow2;
reg [31:0] x1, x2, x3, x4;
reg rdy1, rdy2;

always @(posedge clk) begin
    if (rst) begin
        y    <= 0;
        rdy  <= 1;
        pow2 <= 0;
        pow3 <= 0;
        pow4 <= 0;
        pow5 <= 0;
        x1   <= 0;
        x2   <= 0;
        x3   <= 0;
        x4   <= 0;
        rdy1 <= 0;
        rdy2 <= 0;

        temp_pow2 <= 0;
    end else begin

        if (start) begin
            rdy  <= 0;
            rdy1 <= 1;
        end

        rdy2 <= rdy1;
        rdy  <= rdy2;

        x1   <= x;
        x2   <= x1;
        x3   <= x2;
        x4   <= x3;

        pow2      <= x * x;
        pow3      <= pow2 * x1; // x ^ 3
        pow4      <= pow3 * x2; // x ^ 4
        pow5      <= pow4 * x3; // x ^ 5
        temp_pow2  = x4 * x4;   // x4 ^ 2
        y         <= pow5 + temp_pow2; // x ^ 5 + x ^ 2
    end
end
    
endmodule