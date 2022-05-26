module cubic (
    input clk_i,
    input rst_i,
    input start_i,
    input [31:0] x_bi,
    
    output reg [63:0] y_bo,
    output reg busy_o,
    output reg [2:0] state_bo
);

localparam IDLE = 2'b00;
localparam WORK = 2'b01;
localparam END  = 2'b10;
parameter  SIZE = 8;

reg [2:0] state;
reg [31:0] x;
reg [63:0] b;
reg [63:0] y;
reg signed [3:0] ctr;

always @(posedge clk_i) begin
    if (rst_i) begin
        // reset logic
        state <= IDLE;
        state_bo <= 0;
        x     <= 0;
        ctr   <= 0;
        b     <= 0;
        busy_o <= 0;
        y_bo   <= 0;
        y      <= 0;
    end else begin
        case (state)
            IDLE:begin
                if (start_i) begin
                    state <= WORK;
                    state_bo <= 1;
                    x     <= x_bi;
                    ctr   <= SIZE - 2;
                    busy_o <= 1;
                    y      <= 0;
                end
            end
            
            WORK:begin
                if (ctr < 0) begin
                    state <= END;
                    state_bo <= 2;
                end else begin
                    y = y << 1;
                    b = (3 * y * (y + 1) + 1) << ctr;
                    if (x >= b) begin
                        x <= x - b;
                        y <= y + 1;
                        ctr <= ctr - 3;
                    end else begin
                        ctr <= ctr - 3;
                    end 
                end
            end
 
            END:begin
                y_bo <= y;
                busy_o <= 0;
                state  <= 3;
                state_bo <= 3;
            end

            3:begin
                y_bo <= y_bo;
                busy_o <= busy_o;
                state <= 3;
                state_bo <= 3;
            end
            
            default:begin
                state <= IDLE;
                state_bo <= 0;
            end
        endcase
    end
end
    
endmodule 