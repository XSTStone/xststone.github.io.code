module slave(
    input HCLK_i,
    input HRESETn_i,
    output [31:0] HRDATA_bo,

    input [31:0] HADDR_bi,
    input [31:0] HWDATA_bi,
    input HWRITE_i,

    input [2:0] CS_bi,
    
    output sclk_o,
    output mosi_o,
    input  miso_i,
    output cs_o
);

parameter BASE_ADDR = 0;
parameter CLK_DIV   = 10;


wire [31:0] data_rx, data_tx;
wire data_rx_wr, data_tx_wr;
wire busy;

spi_master  #(.CLK_DIV(CLK_DIV))
spimaster (
    .clk_i(HCLK_i),
    .rst_i(!HRESETn_i),

    .sclk_o(sclk_o),
    .mosi_o(mosi_o),
    .miso_i(miso_i),
    .cs_o(cs_o),

    .data_rx_bo(data_rx),
    .data_rx_wr_o(data_rx_wr),
    .busy_o(busy),

    .data_tx_bi(data_tx),
    .data_tx_wr_i(data_tx_wr)
);

bus_slave #(.BASE_ADDR(BASE_ADDR))
bus_ctrl (
    .HCLK_i(HCLK_i),
    .HRESETn_i(HRESETn_i),
    .HRDATA_bo(HRDATA_bo),

    .HADDR_bi(HADDR_bi),
    .HWDATA_bi(HWDATA_bi),
    .HWRITE_i(HWRITE_i),

    .CS_bi(CS_bi),

    .data_rx_bi(data_rx),
    .data_rx_wr_i(data_rx_wr),
    .busy_i(busy),
    .data_tx_bo(data_tx),
    .data_tx_wr_o(data_tx_wr)
);


endmodule
