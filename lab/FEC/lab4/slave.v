module slave(
    input HCLK_i,
    input HRESETn_i,
    output reg [31:0] HRDATA_bo,

    input [31:0] HADDR_bi,
    input [31:0] HWDATA_bi,
    input HWRITE_i

);

reg [31:0] mem [255:0];

always@(posedge HCLK_i)
    if(HRESETn_i == 0)
        HRDATA_bo <= 0;
    else begin
        if(HWRITE_i)
            mem[HADDR_bi] <= HWDATA_bi;
        else
            HRDATA_bo <= mem[HADDR_bi];
    end


endmodule
