module bus(
    input  [31:0] HADDR_MASTER_bi,
    output [31:0] HADDR_SLAVE_1_bo,
    output [31:0] HADDR_SLAVE_2_bo,
    output [31:0] HADDR_SLAVE_3_bo,
    output [31:0] HADDR_SLAVE_4_bo,

    input  [31:0] HWDATA_MASTER_bi,
    output [31:0] HWDATA_SLAVE_1_bo,
    output [31:0] HWDATA_SLAVE_2_bo,
    output [31:0] HWDATA_SLAVE_3_bo,
    output [31:0] HWDATA_SLAVE_4_bo,

    input  HWRITE_MASTER_i,
    output reg HWRITE_SLAVE_1_o,
    output reg HWRITE_SLAVE_2_o,
    output reg HWRITE_SLAVE_3_o,
    output reg HWRITE_SLAVE_4_o,

    output reg [31:0] HRDATA_MASTER_bo,
    input  [31:0] HRDATA_SLAVE_1_bi,
    input  [31:0] HRDATA_SLAVE_2_bi,
    input  [31:0] HRDATA_SLAVE_3_bi,
    input  [31:0] HRDATA_SLAVE_4_bi
);

assign HADDR_SLAVE_1_bo = HADDR_MASTER_bi;
assign HADDR_SLAVE_2_bo = HADDR_MASTER_bi;
assign HADDR_SLAVE_3_bo = HADDR_MASTER_bi;
assign HADDR_SLAVE_4_bo = HADDR_MASTER_bi;

assign HWDATA_SLAVE_1_bo = HWDATA_MASTER_bi;
assign HWDATA_SLAVE_2_bo = HWDATA_MASTER_bi;
assign HWDATA_SLAVE_3_bo = HWDATA_MASTER_bi;
assign HWDATA_SLAVE_4_bo = HWDATA_MASTER_bi;

always@*
    if(HADDR_MASTER_bi < 32'h20) begin
        HWRITE_SLAVE_1_o <= HWRITE_MASTER_i;
        HRDATA_MASTER_bo <= HRDATA_SLAVE_1_bi;
    end else if(HADDR_MASTER_bi < 32'h30 ) begin
        HWRITE_SLAVE_2_o <= HWRITE_MASTER_i;
        HRDATA_MASTER_bo <= HRDATA_SLAVE_2_bi;
    end else if(HADDR_MASTER_bi < 32'h40 ) begin
        HWRITE_SLAVE_3_o <= HWRITE_MASTER_i;
        HRDATA_MASTER_bo <= HRDATA_SLAVE_3_bi;
    end else begin
        HWRITE_SLAVE_4_o <= HWRITE_MASTER_i;
        HRDATA_MASTER_bo <= HRDATA_SLAVE_4_bi;
    end
endmodule
