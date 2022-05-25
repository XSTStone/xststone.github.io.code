module mult (
    input clk_i,
    input rst_i, // _i means input 
    input start_i,

    input [31:0] a_bi, // _bi means input with bus. Bus means several digital lines

    output reg [63:0] y_bo, // _bo means output with bus. Bus means several digital lines
    output reg busy_o // _o means output
);

localparam IDLE = 2'b00;
localparam WORK = 2'b01;
localparam END  = 2'b10;
parameter  SIZE = 8;

reg [1:0] state;
reg [31:0] a;
reg [2:0] ctr;
reg [63:0] part_res;
wire [63:0] shifted_part_sum;
wire [63:0] part_sum;

assign part_sum = a & {64{a[ctr]}};
assign shifted_part_sum = part_sum << ctr;

always @(posedge clk_i) begin
    if (rst_i) begin
        //reset logic
        ctr      <= 0;
        busy_o   <= 0;
        part_res <= 0;
        y_bo     <= 0;
        state    <= IDLE;
    end else begin
        case (state)
            IDLE:begin
                if (start_i) begin  
                    state    <= WORK;
                    a        <= a_bi;
                    ctr      <= 0;
                    part_res <= 0;
                    busy_o   <= 1;
                end
            end

            WORK:begin
                if (ctr == SIZE - 1) begin
                    state <= END;
                end
                part_res  <= part_res + shifted_part_sum;
                ctr       <= ctr + 1;
            end

            END:begin
                y_bo   <= part_res;
                busy_o <= 0;
                state  <= IDLE;
            end

            default:begin
                state  <= IDLE;
            end
        endcase
    end

end

endmodule