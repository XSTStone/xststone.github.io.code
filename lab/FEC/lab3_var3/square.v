module square (
    input clk_i,
    input rst_i,  
    input start_i,

    input [31:0] a_bi, b_bi,

    output reg [63:0] y_a_bo,
    output reg [63:0] y_b_bo,
    output reg [2:0] state_o // 0 -> ready 1 -> work 2 -> wait
);

parameter SIZE = 32;

reg [2:0] state;
reg [31:0] a, b;
reg [5:0] ctr;
reg [63:0] part_res_a, part_res_b;
wire [63:0] shifted_part_sum_a, shifted_part_sum_b;
wire [63:0] part_sum_a, part_sum_b;

assign part_sum_a = a & {64{a[ctr]}};
assign part_sum_b = b & {64{b[ctr]}};
assign shifted_part_sum_a = part_sum_a << ctr;
assign shifted_part_sum_b = part_sum_b << ctr;

always @(posedge clk_i) begin
    if (rst_i) begin
        ctr <= 0;
        state <= 0;
        part_res_a <= 0;
        part_res_b <= 0;

        state_o <= 0;
        y_a_bo <= 0;
        y_b_bo <= 0;
    end else begin
        case (state)
            0:begin
                if (start_i) begin
                    state <= 1;
                    a <= a_bi;
                    b <= b_bi;
                    ctr <= 0;
                    part_res_a <= 0;
                    part_res_b <= 0;
                    state_o <= 1;
                end
            end

            1:begin
                if (ctr == SIZE - 1) begin
                    state <= 2;
                end
                part_res_a <= part_res_a + shifted_part_sum_a;
                part_res_b <= part_res_b + shifted_part_sum_b;
                ctr <= ctr + 1;
            end

            2:begin
                y_a_bo <= part_res_a;
                y_b_bo <= part_res_b;
                state <= 3;
                state_o <= 2;
            end

            3:begin
                state <= 3;
                state_o <= 2;
                y_a_bo <= y_a_bo;
                y_b_bo <= y_b_bo;
            end

            default:begin
                state <= 0;
                state_o <= 0;
            end
        endcase
    end
end

endmodule