module root (
    input clk_i,
    input rst_i,
    input start_i,

    input [31:0] x_bi,

    output reg [31:0] y_bo,
    output reg [2:0] state_o // 0 -> ready 1 -> work 2-> wait
);

parameter SIZE = 32;

reg [31:0] m;
reg [31:0] x;
reg [3:0] state; // 0 1 2 3
reg [31:0] b;
reg [31:0] y;


always @(posedge clk_i) begin
    if (rst_i) begin
        state <= 0;
        m <= 0;
        b <= 0;
        y <= 0;
        x <= 0;
        y_bo <= 0;
        state_o <= 0;
    end else begin
        case (state)
            0:begin
                if (start_i) begin
                    state <= 1;
                    m <= 1 << (SIZE - 2);
                    state_o <= 1;
                    x <= x_bi;
                end
            end

            1:begin
                if (m == 0) begin
                    state <= 2;
                end else begin
                    b = y | m;
                    y = y >> 1;
                    if (x >= b) begin
                        x <= x - b;
                        y <= y | m;
                        m <= m >> 2;
                    end else begin
                        m <= m >> 2;
                    end
                end
            end

            2:begin
                state <= 3;
                y_bo <= y;
            end

            3:begin
                state_o <= 2;
                state <= 3;
                y_bo <= y_bo;
            end

            default:begin
                state <= 0;
            end
        endcase
    end
end

endmodule
