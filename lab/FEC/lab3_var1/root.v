module root (
    input clk_i,
    input rst_i,
    input start_i,

    input [31:0] x_bi,
    
    output reg [63:0] y_bo,
    output reg busy_o
);

localparam IDLE = 2'b00;
localparam WORK = 2'b01;
localparam END  = 2'b10;
parameter  SIZE = 8;

reg [1:0] state;
reg [31:0] x;
reg [63:0] b;
reg [63:0] y;
reg signed [3:0] ctr;

always @(posedge clk_i) begin
    if (rst_i) begin
        // reset logic
        state <= IDLE;
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
                    x     <= x_bi;
                    ctr   <= SIZE - 2;
                    busy_o <= 1;
                    y      <= 0;
                end
            end
            
            WORK:begin
                if (ctr < 0) begin
                    state <= END;
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
                state  <= IDLE;
            end
            
            default:begin
                state <= IDLE;
            end
        endcase
    end
end
    
endmodule 