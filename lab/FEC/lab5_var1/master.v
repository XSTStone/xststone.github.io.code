module master(
    input HCLK_i,
    input HRESETn_i,
    input [31:0] HRDATA_bi,

    output reg [31:0] HADDR_bo,
    output reg [31:0] HWDATA_bo,
    output reg [2:0] CS_bo,
    output reg HWRITE_o

);

localparam STATUS_ADDR = 8'h0;
localparam OP_ADDR     = 8'h1;

task write(
    input [31:0] addr,
    input [31:0] data
);
    begin
        @(posedge HCLK_i);
        HADDR_bo <= addr;
        HWRITE_o <= 1;
        if (addr < 32'h20) 
            CS_bo <= 1;
        else if (addr < 32'h30 && addr >= 32'h20) 
            CS_bo <= 2;
        else if (addr < 32'h40 && addr >= 32'h30)
            CS_bo <= 3;
        else if (addr < 32'h50 && addr >= 32'h40) 
            CS_bo <= 4;
        
        @(posedge HCLK_i);
        // $display("=============");
        // $display("In master file, CS_bo == %h", CS_bo);
        HWDATA_bo <= data;
        HWRITE_o <= 1;
        @(posedge HCLK_i);
        @(posedge HCLK_i);
        @(posedge HCLK_i);
        HWRITE_o <= 0;

    end
endtask

task read(
    input  [31:0] addr,
    output [31:0] data
);
    begin
        @(posedge HCLK_i);
        HADDR_bo <= addr;
        HWRITE_o <= 0;

        @(posedge HCLK_i);
        @(posedge HCLK_i);
        data <= HRDATA_bi;
        @(posedge HCLK_i);
        // $display("data is %h", data);
    end


endtask

task wait_ready();

begin

    begin : wait_busy
        while(1) begin
            read(32'h20, data);
            if(data == 1) begin
                // $display("wait_busy done");
                disable wait_busy;
            end
        end
    end
    
    begin : wait_ready
        while(1) begin
            read(32'h20, data);
            if(data == 0) begin
                // $display("wait ready done");
                disable wait_ready;
            end
        end
    end
end

endtask

reg [31:0] data = 1;

initial begin

    write(32'h21, {8'h0, STATUS_ADDR, 16'h1});
    // $display("First step done");
    wait_ready;
    // $display("Waiting done");
    write(32'h21, {8'h0, OP_ADDR, 8'h2, 8'h8}); // a2 b8
    // $display("Second step done");
    wait_ready;
    write(32'h21, 0);
    wait_ready;
    write(32'h21, 0);
    wait_ready;
    // $display("All writes done");
    read(32'h22, data);
    $display("INPUT DATA | A: %h, B: %h", 8'h2, 8'h8);
    $display("READ DATA  | data: %h", data);
    
    write(32'h21, {8'h0, OP_ADDR, 8'h2, 8'h4}); // a2 b4
    wait_ready;
    write(32'h21, 0);
    wait_ready;
    write(32'h21, 0);
    wait_ready;
    read(32'h22, data);
    $display("INPUT DATA | A: %h, B: %h", 8'h2, 8'h4);
    $display("READ DATA  | data: %h", data);

    $finish;

end

endmodule
