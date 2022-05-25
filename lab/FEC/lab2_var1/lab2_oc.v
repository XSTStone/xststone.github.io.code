module lab2_oc (
    input clk,
    input rst,
    input [31:0] x,
    output reg [31:0] y
);
    
always @(posedge clk) begin
    if (rst)
        y <= 0;
    else 
        y <= x * x + x * x * x * x * x;
end

endmodule