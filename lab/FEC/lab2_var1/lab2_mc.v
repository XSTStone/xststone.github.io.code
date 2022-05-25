module lab2_mc (
    input clk,
    input rst,
    input start,
    input [31:0] x,

    output reg rdy,
    output reg [31:0] y_output
);

reg [31:0] y;
reg [31:0] x_buf;
reg [31:0] y_square;
reg [2:0] ctr;
reg [1:0] state; // 0 -> in power 2; 1 -> in power 5;
reg [2:0] current; // 0 -> multiplying; 1 -> x^2 done; 2 -> x^5 done

wire [2:0] ctr_next = (rdy)? 0: ctr + 1;
wire [31:0] y_next  = (rdy)? 1: y * x_buf;

always @(posedge clk) 
    begin
    if (rst) 
        begin
        y_output <= 0;
        y_square <= 0;
        y        <= 0;
        rdy      <= 1;
        x_buf    <= 0;
        ctr      <= 0;
        current  <= 0;
        state    <= 0;
        end 
    else 
        begin
        ctr <= ctr_next;

        if (start) 
            begin
            y        <= 1;
            rdy      <= 0;
            x_buf    <= x;
            y_square <= 0;
            state    <= 0;
            end

        if (!rdy) 
            begin
            if (ctr == 1 && state == 0)
                current <= 1;
            else if (ctr == 3 && state == 1)
                current <= 2;
            else 
                current <= 0;

            if (current == 1) 
                begin
                y_square <= y;
                y        <= 1;
                state    <= 1;
                ctr      <= 0;
                end 

            if (current == 2)
                begin
                y         = y_next + y_square;
                y_output <= y;
                rdy      <= 1;
                end

            if (current == 0)
                begin
                y        <= y_next; 
                end
            end
        end
    end

endmodule