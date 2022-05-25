module spi_master (
    input clk_i,
    input rst_i,

    output reg sclk_o,
    output reg mosi_o,
    input  miso_i,
    output reg cs_o,

    output reg [31:0] data_rx_bo,
    output reg data_rx_wr_o,
    output reg busy_o,

    input [31:0] data_tx_bi,
    input data_tx_wr_i
);

parameter CLK_DIV = 10;

localparam IDLE  = 0;
localparam TRANS = 1;

reg [6:0]  ctr_r;
reg [31:0] clk_div_r;
reg [1:0]  state_r;
reg [31:0] data_rx_buf_r;
reg [31:0] data_tx_buf_r;

wire sclk_posedge = (clk_div_r == CLK_DIV) & (sclk_o == 0);

always@(posedge clk_i)
    if(rst_i) begin
        clk_div_r     <= 0;
        ctr_r         <= 0;
        data_rx_buf_r <= 0;
        sclk_o        <= 0;
    end else begin

        if(busy_o) begin
            clk_div_r <= clk_div_r + 1;

            if(clk_div_r == CLK_DIV) begin
                sclk_o <= ~sclk_o;
                clk_div_r <= 0;
                if(sclk_o) begin
                    ctr_r         <= ctr_r + 1;
                    data_rx_buf_r <= {data_rx_buf_r[30:0], miso_i};
                end
            end
        end
    end
    
always@(posedge clk_i)
    if(rst_i) begin
        data_tx_buf_r <= 0;
        mosi_o <= 0;
    end else begin
        if(data_tx_wr_i)
            data_tx_buf_r <= data_tx_bi;
            
        if(sclk_posedge) begin 
            mosi_o <= data_tx_buf_r[31];
            data_tx_buf_r <= data_tx_buf_r << 1;
        end
    end

always@(posedge clk_i)
    if(rst_i) begin
        state_r <= 0;
        busy_o  <= 0;
        cs_o    <= 1;
        data_rx_wr_o  <= 0;
    end else begin
        case(state_r)
            IDLE: 
                begin
                    data_rx_wr_o  <= 0;
                    if(data_tx_wr_i) begin
                        busy_o <= 1;
                        state_r <= TRANS;
                        cs_o  <= 0;
                        ctr_r <= 0;
                        
                    end
                end
            TRANS: 
                begin

                    if(ctr_r == 6'h20) begin
                        cs_o    <= 1;
                        busy_o  <= 0;
                        ctr_r   <= 0;
                        state_r <= IDLE;
                        data_rx_wr_o  <= 1;
                        data_rx_bo <= data_rx_buf_r;
                    end
                end
            default:
                state_r <= IDLE;
        endcase
    end

endmodule
