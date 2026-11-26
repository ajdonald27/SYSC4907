// RATS: AXI4-Stream pass-through that re-generates TLAST every FRAME_LEN beats.
//
// The Xilinx FFT raises TLAST at the end of every 128-sample transform. The AXI DMA's
// S2MM channel ends a transfer at the first TLAST it sees, so without this block a
// 32768-word frame would stop after 128 words. This block ignores the incoming TLAST and
// raises its own on beat FRAME_LEN-1 (default: one S_system frame = 256 x 128 = 32768).
// Data, valid and ready pass straight through (no added latency).

`timescale 1ns / 1ps

module axis_tlast_gen #(
    parameter integer DATA_W    = 32,
    parameter integer FRAME_LEN = 32768
) (
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET aresetn" *)
    input  wire              aclk,
    input  wire              aresetn,

    input  wire [DATA_W-1:0] s_axis_tdata,
    input  wire              s_axis_tvalid,
    output wire              s_axis_tready,
    input  wire              s_axis_tlast,     // ignored on purpose

    output wire [DATA_W-1:0] m_axis_tdata,
    output wire              m_axis_tvalid,
    input  wire              m_axis_tready,
    output wire              m_axis_tlast
);

    localparam integer CW = (FRAME_LEN > 1) ? $clog2(FRAME_LEN) : 1;

    reg [CW-1:0] beat;

    assign m_axis_tdata  = s_axis_tdata;
    assign m_axis_tvalid = s_axis_tvalid;
    assign s_axis_tready = m_axis_tready;
    assign m_axis_tlast  = (beat == FRAME_LEN - 1);

    always @(posedge aclk) begin
        if (!aresetn)
            beat <= {CW{1'b0}};
        else if (s_axis_tvalid && m_axis_tready)
            beat <= (beat == FRAME_LEN - 1) ? {CW{1'b0}} : beat + 1'b1;
    end

endmodule
