module periph_dev (    
    input  sclk_i,
    input  mosi_i,
    output reg miso_o,
    input  cs_i
);

localparam RESET_ADDR    = 8'h0;
localparam OP_ADDR  = 8'h1;


reg [7:0]  addr_r, cmd_r;

reg [31:0] rx_data_buf_r, tx_data_buf_r; 

reg start_r;

reg [7:0] a_r, b_r;

wire [15:0] y;
wire busy;

reg rst_r;

final calc_unit(
    .clk_i(sclk_i),
    .rst_i(rst_r),

    .start_i(start_r),

    .a_bi(a_r),
    .b_bi(b_r),

    .y_bo(y), 
    .busy_o(busy)
);

localparam STATE0 = 0;
localparam STATE1 = 1;

reg state_r = STATE0;
reg [6:0] ctr_r = 0;

wire [31:0] rx_data_buf_next = {rx_data_buf_r[30:0], mosi_i};

localparam ADDR_CTR_VAL = 15;
localparam DATA_CTR_VAL  = 30;


always@(posedge sclk_i) begin
    miso_o        <= tx_data_buf_r[31];
    
    if(state_r == STATE0) begin
        tx_data_buf_r <= {14'h0, busy, y, 1'b0};
        miso_o        <= 0;
    end else
        tx_data_buf_r <= tx_data_buf_r << 1;
end


always@(negedge sclk_i)

        if(cs_i == 0) begin
            case(state_r)
                STATE0:
                    begin
                        start_r       <= 0;
                        ctr_r         <= 0;
                        rst_r         <= 0;
                        state_r       <= STATE1;
                        ctr_r         <= 0;
                        rx_data_buf_r <= rx_data_buf_next;
                        
                    end
                STATE1:
                    begin
                        ctr_r         <= ctr_r + 1;
                        rx_data_buf_r <= rx_data_buf_next;
                        
                        
                        case(ctr_r)
                                ADDR_CTR_VAL: 
                                    begin
                                        addr_r <= rx_data_buf_r[7:0];
                                    end
                                DATA_CTR_VAL: 
                                    begin
                                        state_r <= STATE0;
                                        case(addr_r)
                                            OP_ADDR:
                                                if(busy == 0) begin
                                                    a_r <= rx_data_buf_next[15:8];
                                                    b_r <= rx_data_buf_next[7:0];
                                                    start_r <= 1; 
                                                end
                                            RESET_ADDR:
                                                rst_r <= rx_data_buf_next[0];
                                        endcase
                                    end
                            endcase
                    end
            endcase    
        end    
       

endmodule
 
