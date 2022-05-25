module final (
    input clk_i,
    input rst_i,
    input start_i,

    input [31:0] a_bi,
    input [31:0] b_bi,

    output reg [63:0] y_bo,
    output reg busy_o
);

localparam IDLE = 2'b00;
localparam WORK = 2'b00;
localparam END  = 2'b00;
parameter SIZE  = 32;

reg [2:0] mult_state; // 0 -> IDLE 1 -> WORK 2 -> END 3 -> WAIT
reg [31:0] calc_a; // copy of input value a
reg [63:0] mult_y;
reg [5:0] mult_ctr;
reg [63:0] part_res;
reg [1:0] is_end;
wire [63:0] shifted_part_sum;
wire [63:0] part_sum;

assign part_sum = calc_a & {64{calc_a[mult_ctr]}};
assign shifted_part_sum = part_sum << mult_ctr;
    
reg [2:0] root_state; // 0 -> IDLE 1 -> WORK 2 -> END 3 -> WAIT
reg [31:0] calc_b; // copy of input value b
reg [63:0] calc_y; // value y during calculating process
reg [63:0] root_b; // value b during calculating process
reg [63:0] root_y; // final value y for output 
reg signed [5:0] root_ctr;

always @(posedge clk_i) begin
    if (rst_i) begin
        mult_state <= IDLE;
        calc_a <= 0;
        mult_y <= 0;
        mult_ctr <= 0;
        part_res <= 0;

        root_state <= IDLE;
        calc_b <= 0;
        calc_y <= 0;
        root_b <= 0;
        root_y <= 0;
        root_ctr <= 0;
    end else begin
        // mult logic
        case (mult_state)
            0:begin
                if (start_i) begin
                    mult_state <= 1;
                    calc_a <= a_bi;
                    mult_ctr <= 0;
                    part_res <= 0;
                    busy_o <= 1;
                end
            end 

            1:begin
                if (mult_ctr == SIZE - 1) begin
                    mult_state <= 2;
                end
                part_res <= part_res + shifted_part_sum;
                mult_ctr <= mult_ctr + 1;
            end

            2:begin
                mult_y <= part_res;
                mult_state <= 3;
            end
            
            3:begin
                mult_state <= 3;
                mult_y <= mult_y;
            end

            default:begin
                mult_state <= 0;
            end
        endcase

        case (root_state)
            0:begin
                if (start_i) begin
                    root_state <= 1;
                    // root_b <= b_bi;
                    calc_b <= b_bi;
                    root_ctr <= SIZE - 2;
                    busy_o <= 1;
                    root_y <= 0;
                    calc_y <= 0;
                end
            end

            1:begin
                if (root_ctr < 0) begin
                    root_state <= 2;
                end else begin
                    calc_y = calc_y << 1;
                    root_b = (3 * calc_y * (calc_y + 1) + 1) << root_ctr;
                    if (calc_b >= root_b) begin
                        calc_b <= calc_b - root_b;
                        calc_y <= calc_y + 1;
                        root_ctr <= root_ctr - 3;
                    end else begin
                        root_ctr <= root_ctr - 3; 
                    end
                end
            end

            2:begin
                root_y <= calc_y;
                root_state <= 3;
            end

            3:begin
                root_state <= 3;
                root_y <= root_y;
            end

            default:begin
                root_state <= 0;
            end
        endcase

        if (root_state == 3 && mult_state == 3) begin
            root_state <= 0;
            mult_state <= 0;
            is_end <= 1;
            y_bo <= root_y + mult_y;
            busy_o <= 0;
        end
    end
end

endmodule