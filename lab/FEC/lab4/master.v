module master(
    input HCLK_i,
    input HRESETn_i,
    input [31:0] HRDATA_bi,

    output reg [31:0] HADDR_bo,
    output reg [31:0] HWDATA_bo,
    output reg HWRITE_o

);

task write(
    input [31:0] addr,
    input [31:0] data
);
    begin
        @(posedge HCLK_i);
        HADDR_bo  <= addr;
		HWDATA_bo <= data;
        HWRITE_o  <= 1;
        
        @(posedge HCLK_i);
        @(posedge HCLK_i);
        HWRITE_o <= 0;

    end
endtask

task read(
    input [31:0] addr
);
    begin
        @(posedge HCLK_i);
        HADDR_bo <= addr;
        HWRITE_o <= 0;

        @(posedge HCLK_i);
        @(posedge HCLK_i);
        $display("READ DATA | addr: %h, data: %h", addr, HRDATA_bi);
    end


endtask

initial begin
    write(32'h15, 32'h55AA);
    read(32'h1A);
    read(32'h15);

    write(32'h30, 32'hAA55);
    read(32'h31);
    read(32'h30);

    $finish;

end
endmodule
