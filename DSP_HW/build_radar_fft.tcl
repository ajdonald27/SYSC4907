# RATS: builds the PYNQ-Z2 overlay for the range FFT, from nothing to a bitstream.
#
#   PS7 (100 MHz FCLK0) --GP0--> AXI DMA (control) + AXI GPIO (FFT config word)
#   DMA MM2S --> Xilinx FFT (128-pt, scaled, pipelined, natural order) --> axis_tlast_gen --> DMA S2MM
#   DMA memory ports --> PS7 HP0
#
# Usage (Vivado Tcl Console, or "Vivado 2024.1 Tcl Shell" with: vivado -mode batch -source build_radar_fft.tcl):
#   cd C:/radar/fpga
#   source build_radar_fft.tcl
# For the bring-up design (DMA -> FIFO -> DMA) type `set MODE loopback` before sourcing.
# Output: out/radar_fft.bit + out/radar_fft.hwh (or radar_loopback.*). Copy both to the PYNQ.

if {![info exists MODE]} { set MODE fft }   ;# fft | loopback
set NAME     [expr {$MODE eq "fft" ? "radar_fft" : "radar_loopback"}]
set JOBS     4
set FRAME_LEN 32768       ;# 256 chirps x 128 samples (S_system)

set here [file dirname [file normalize [info script]]]
set proj_dir "$here/vivado_$NAME"
set out_dir  "$here/out"
file mkdir $out_dir

# --- helper: newest installed version of an IP, so the script works across Vivado versions ---
proc ipv {name} {
    set defs [lsort [get_ipdefs -all "xilinx.com:ip:${name}:*"]]
    if {[llength $defs] == 0} { error "IP $name not found in this Vivado install" }
    return [lindex $defs end]
}

# --- project ---
catch {close_project}   ;# allows building loopback then fft in one Vivado session
create_project $NAME $proj_dir -part xc7z020clg400-1 -force
set board [lindex [get_board_parts -quiet *pynq-z2*] end]
if {$board ne ""} {
    set_property board_part $board [current_project]
    puts "RATS: using board files $board"
} else {
    puts "RATS: PYNQ-Z2 board files not found, using the part only (fine for a PYNQ overlay)"
}
add_files -norecurse "$here/axis_tlast_gen.v"
update_compile_order -fileset sources_1

create_bd_design $NAME

# --- Zynq processing system ---
set ps [create_bd_cell -type ip -vlnv [ipv processing_system7] processing_system7_0]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config [list make_external "FIXED_IO, DDR" apply_board_preset [expr {$board ne "" ? 1 : 0}] Master "Disable" Slave "Disable"] $ps
set_property -dict [list \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
] $ps

# --- AXI DMA, simple mode (no scatter-gather), 32-bit streams ---
set dma [create_bd_cell -type ip -vlnv [ipv axi_dma] axi_dma_0]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {23} \
    CONFIG.c_m_axi_mm2s_data_width {32} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_m_axi_s2mm_data_width {32} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s2mm_burst_size {16} \
] $dma

# --- AXI connections (control on GP0, memory on HP0), clocks and resets by automation ---
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_dma_0/S_AXI_LITE} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}] \
    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/axi_dma_0/M_AXI_MM2S} Slave {/processing_system7_0/S_AXI_HP0} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}] \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/axi_dma_0/M_AXI_S2MM} Slave {/processing_system7_0/S_AXI_HP0} ddr_seg {Auto} intc_ip {Auto} master_apm {0}] \
    [get_bd_intf_pins axi_dma_0/M_AXI_S2MM]

set clk  [get_bd_pins processing_system7_0/FCLK_CLK0]
set rstc [lindex [get_bd_cells -filter {VLNV =~ "*proc_sys_reset*"}] 0]
set rstn [get_bd_pins $rstc/peripheral_aresetn]

if {$MODE eq "loopback"} {
    # --- bring-up: DMA -> FIFO -> DMA ---
    set fifo [create_bd_cell -type ip -vlnv [ipv axis_data_fifo] axis_data_fifo_0]
    connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins axis_data_fifo_0/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins axis_data_fifo_0/M_AXIS] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
    connect_bd_net $clk  [get_bd_pins axis_data_fifo_0/s_axis_aclk]
    connect_bd_net $rstn [get_bd_pins axis_data_fifo_0/s_axis_aresetn]
} else {
    # --- the range FFT ---
    set fft [create_bd_cell -type ip -vlnv [ipv xfft] xfft_0]
    set_property -dict [list \
        CONFIG.transform_length {128} \
        CONFIG.implementation_options {pipelined_streaming_io} \
        CONFIG.target_clock_frequency {100} \
        CONFIG.data_format {fixed_point} \
        CONFIG.input_width {16} \
        CONFIG.phase_factor_width {16} \
        CONFIG.scaling_options {scaled} \
        CONFIG.rounding_modes {convergent_rounding} \
        CONFIG.output_ordering {natural_order} \
        CONFIG.throttle_scheme {nonrealtime} \
        CONFIG.aresetn {true} \
    ] $fft

    # TLAST every FRAME_LEN beats so the DMA receives one whole frame per transfer
    set tg [create_bd_cell -type module -reference axis_tlast_gen axis_tlast_gen_0]
    set_property CONFIG.FRAME_LEN $FRAME_LEN $tg

    connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]    [get_bd_intf_pins xfft_0/S_AXIS_DATA]
    connect_bd_intf_net [get_bd_intf_pins xfft_0/M_AXIS_DATA]       [get_bd_intf_pins axis_tlast_gen_0/s_axis]
    connect_bd_intf_net [get_bd_intf_pins axis_tlast_gen_0/m_axis]  [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
    connect_bd_net $clk  [get_bd_pins xfft_0/aclk] [get_bd_pins axis_tlast_gen_0/aclk]
    connect_bd_net $rstn [get_bd_pins xfft_0/aresetn] [get_bd_pins axis_tlast_gen_0/aresetn]

    # FFT config word (FWD_INV + scaling schedule) from an AXI GPIO, so Python sets the scaling
    set cfg_w [expr {[get_property LEFT [get_bd_pins xfft_0/s_axis_config_tdata]] + 1}]
    puts "RATS: FFT config word is $cfg_w bits"
    set gpio [create_bd_cell -type ip -vlnv [ipv axi_gpio] axi_gpio_0]
    set_property -dict [list CONFIG.C_GPIO_WIDTH $cfg_w CONFIG.C_ALL_OUTPUTS {1}] $gpio
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
        Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_gpio_0/S_AXI} ddr_seg {Auto} intc_ip {Auto} master_apm {0}] \
        [get_bd_intf_pins axi_gpio_0/S_AXI]
    connect_bd_net [get_bd_pins axi_gpio_0/gpio_io_o] [get_bd_pins xfft_0/s_axis_config_tdata]
    set one [create_bd_cell -type ip -vlnv [ipv xlconstant] cfg_valid]
    set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {1}] $one
    connect_bd_net [get_bd_pins cfg_valid/dout] [get_bd_pins xfft_0/s_axis_config_tvalid]
}

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

# --- wrapper, synthesis, implementation, bitstream ---
set bd_file [get_files "$NAME.bd"]
make_wrapper -files $bd_file -top
add_files -norecurse [glob "$proj_dir/$NAME.gen/sources_1/bd/$NAME/hdl/${NAME}_wrapper.v"]
set_property top ${NAME}_wrapper [current_fileset]
update_compile_order -fileset sources_1

set t0 [clock seconds]
launch_runs impl_1 -to_step write_bitstream -jobs $JOBS
wait_on_run impl_1
set mins [expr {([clock seconds] - $t0) / 60.0}]
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "RATS: implementation did not finish; open the project and check the Messages tab"
}

# --- collect the two files PYNQ needs, with the same base name ---
file copy -force "$proj_dir/$NAME.runs/impl_1/${NAME}_wrapper.bit" "$out_dir/$NAME.bit"
file copy -force [lindex [glob "$proj_dir/$NAME.gen/sources_1/bd/$NAME/hw_handoff/$NAME.hwh"] 0] "$out_dir/$NAME.hwh"
puts [format "RATS: done in %.1f min -> %s/%s.bit and %s.hwh" $mins $out_dir $NAME $NAME]
