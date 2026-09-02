Team Members Roles & Responsbilities: 
- Initial Draft : 07/21/2026
- Lastest Rev: 8/8/2026

- This document outlines the roles & responsbilities + deliverables (roughly) for each team member. This list is both flexible and subject to change as we refine the scope of the project. 

Team Members 1 & 2 (AJ Donald) & (Jayven Larsen) : Hardware, FPGA & Radar Processing
------------------------------------------------------------------------------
Responsibilities
- Design the overall FPGA hardware architecture
    - Review & Refine block diagram.
- Develop and validate the Python radar reference model
    - Latest reference model 07/21/2026
- Design and implement the radar DSP pipeline (FIR, FFT, peak detection, velocity estimation)
    - Implement the algorithms in Verilog.
e radar hardware, ADCs, and communication interfaces
    - Includes writing SPI driver for ADC, etc.
- Develop FPGA-to-ARM communication and embedded software interfaces
    - Subject to be re-assigned.
- Verify functionality through simulation and hardware testing 
    - Should be done as much as possible in a Vivado project.
- Produce hardware, interface, and verification documentation
    - Produce necessary documentation for rest of team. 


Team Member 3 : Computer Vision & Camera Processing (Andrew Tawfik)
---------------------------------------------------
Responsibilities
- Select and integrate the camera system
    - Potentially use RPi Camera module 
- Develop the OpenCV processing pipeline
- Implement motion detection, object detection, and tracking
    - AJ can provide older motion detection algorithm from SYSC3010
- Investigate advanced object detection methods (stretch goals)
    - Potential use of AI/ML model or NN for Computer Vision


Team Member 4 : Sensor Fusion & Object Tracking (Liam Bennet)
---------------------------------------------------
Responsibilities
- Fuse radar and camera detections
- Develop coordinate transformations and synchronization
- Implement object association and tracking
- Estimate object confidence and maintain target tracks
- Produce a unified object list for downstream visualization
- Develop communication between the vision and FPGA subsystems

Team Member 5 : System Integration & User Interface (Rami Ayoub)
---------------------------------------------------
Responsibilities
- Develop the user interface
- Integrate all project subsystems
- Implement logging, configuration, and data recording
- Coordinate system-level testing and validation
- Benchmark overall system performance
- Prepare the final demonstration and presentation




Potential Deliverables (SUBJECT TO CHANGE BASED ON GROUP/INDIVIDUAL WORK ITEMS)
---------------------------------------------------
Hardware Team Deliverables
- FPGA radar processing pipeline
- Hardware interface drivers
- Verified DSP implementation
- Hardware documentation

Vision Deliverables
- Camera processing pipeline
- Object detection module
- Object tracking module

Fusion Deliverables
- Sensor fusion module
- Unified object list
- Tracking/confidence system

Integration Deliverables
- Real-time GUI
- Integrated software application
- Final demonstration environment