module bus_slave(
    input HCLK_i,
    input HRESETn_i,
    output reg [31:0] HRDATA_bo,

    input [31:0] HADDR_bi,
    input [31:0] HWDATA_bi,
    input HWRITE_i,

    input [2:0] CS_bi,

    input  [31:0] data_rx_bi,
    input  data_rx_wr_i,
    input  busy_i,
    output reg [31:0] data_tx_bo,
    output reg data_tx_wr_o
);

parameter BASE_ADDR = 0;
// parameter SIZE = 32;

localparam STATUS_REG_ADDR_BIAS = 0;
localparam DATA_TX_ADDR_BIAS    = 1;
localparam DATA_RX_ADDR_BIAS    = 2;

reg [7:0]  status_r;
reg [31:0] data_rx_r;
reg [15:0] base_addr;
reg [2:0] xst_is_nb;

always@(posedge HCLK_i)
    if(HRESETn_i == 0) begin
        HRDATA_bo     <= 0;
        data_tx_bo    <= 0;
        data_tx_wr_o  <= 0;
        base_addr     <= 0;
    end else begin
        data_tx_wr_o <= 0;
        base_addr    <= BASE_ADDR + 0 + (CS_bi - 1) * 32;
        // $display("In bus_slave.v file: CS_bi == %h", CS_bi);
        // $display("In bus_slave.v file: base_addr == %h", base_addr);
        // $display("==============");


        if(HWRITE_i)
            case(HADDR_bi)
                DATA_TX_ADDR_BIAS + base_addr:
                    begin
                        // $display("In bus_slave.v file: STATUS_REG_ADDR_BIAS = %h", STATUS_REG_ADDR_BIAS);
                        // $display("In bus_slave.v file: Got data tx addr signal");
                        data_tx_bo   <= HWDATA_bi;
                        data_tx_wr_o <= 1;
                    end
            endcase
        else begin
            // $display("In bus_slave.v file: write signal is false X");
            // $display("In bus_slave.v file: HADDR_bi is %h", HADDR_bi);
            // $display("In bus_slave.v file: STATUS_REG_ADDR is %h", STATUS_REG_ADDR_BIAS + base_addr);
            // $display("In bus_slave.v file: DATA_TX_ADDR is %h", DATA_TX_ADDR_BIAS + base_addr);
            // $display("In bus_slave.v file: DATA_RX_ADDR is %h", DATA_RX_ADDR_BIAS + base_addr);
            case(HADDR_bi)
                STATUS_REG_ADDR_BIAS + base_addr:begin
                    HRDATA_bo <= {24'h0, status_r};
                    // $display("HRDATA_bo is %h", HRDATA_bo);
                    // $display("----------------------");
                    xst_is_nb = 0;
                end
                DATA_TX_ADDR_BIAS + base_addr:begin
                    HRDATA_bo <= data_tx_bo;
                    xst_is_nb = 1;
                end
                DATA_RX_ADDR_BIAS + base_addr:begin
                    HRDATA_bo <= data_rx_r;
                    xst_is_nb = 2;
                end
                default:
                    HRDATA_bo <= 0;
            endcase
        end
    end

always@(posedge HCLK_i)
    if(HRESETn_i == 0) begin
        status_r  <= 0;
        data_rx_r <= 0;
    end else begin
        status_r <= {7'h0, busy_i};
        
        if(data_rx_wr_i)
            data_rx_r <= data_rx_bi;
    end


endmodule
