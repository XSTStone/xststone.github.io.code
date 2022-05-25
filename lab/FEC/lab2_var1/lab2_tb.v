`timescale 1ns/1ps

module lab2_tb;

reg [31:0] x;
reg rst, clk, start;
wire [31:0] y_oc, y_mc, y_pipe;
wire rdy_mc, rdy_pipe;

lab2_oc u_lab2_oc(
    .clk ( clk ),
    .rst ( rst ),
    .x   ( x   ),
    .y   ( y   )
);

// lab2_mc uut_mc(
//     .clk(clk),
//     .rst(rst),
//     .start(start),
//     .x(x),
//     .y_output(y_mc),
//     .rdy(rdy_mc)
// );

lab2_mc u_lab2_mc(
    .clk   ( clk   ),
    .rst   ( rst   ),
    .start ( start ),
    .x     ( x     ),
    .rdy   ( rdy   ),
    .y_output  ( y_mc  )
);

lab2_pipe u_lab2_pipe(
    .clk   ( clk    ),
    .rst   ( rst    ),
    .start ( start  ),
    .x     ( x      ),
    .rdy   ( rdy    ),
    .y     ( y_pipe )
);


always #5 clk = ~clk;

always @(negedge clk) 
    x = ($random % 64) & 8'hff;

initial begin
    
    $dumpfile("time.vcd");
    $dumpvars(0, lab2_tb);

    clk   = 0;
    rst   = 1;
    x     = 0;
    // for mc
    start = 0;

    #20
    rst   = 0;
    // for mc
    start = 1;

    // for mc
    #10
    start = 0;

    // mc
    #100
    start = 1;

    // mc
    #10
    start = 0;

    #100
    $finish;

end

endmodule