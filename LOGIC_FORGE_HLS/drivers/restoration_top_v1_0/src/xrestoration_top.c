// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xrestoration_top.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XRestoration_top_CfgInitialize(XRestoration_top *InstancePtr, XRestoration_top_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XRestoration_top_Start(XRestoration_top *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_AP_CTRL) & 0x80;
    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XRestoration_top_IsDone(XRestoration_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XRestoration_top_IsIdle(XRestoration_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XRestoration_top_IsReady(XRestoration_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XRestoration_top_EnableAutoRestart(XRestoration_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XRestoration_top_DisableAutoRestart(XRestoration_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_AP_CTRL, 0);
}

void XRestoration_top_Set_input_r(XRestoration_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_INPUT_R_DATA, (u32)(Data));
    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_INPUT_R_DATA + 4, (u32)(Data >> 32));
}

u64 XRestoration_top_Get_input_r(XRestoration_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_INPUT_R_DATA);
    Data += (u64)XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_INPUT_R_DATA + 4) << 32;
    return Data;
}

void XRestoration_top_Set_output_r(XRestoration_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_OUTPUT_R_DATA, (u32)(Data));
    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_OUTPUT_R_DATA + 4, (u32)(Data >> 32));
}

u64 XRestoration_top_Get_output_r(XRestoration_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_OUTPUT_R_DATA);
    Data += (u64)XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_OUTPUT_R_DATA + 4) << 32;
    return Data;
}

void XRestoration_top_Set_weights(XRestoration_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_WEIGHTS_DATA, (u32)(Data));
    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_WEIGHTS_DATA + 4, (u32)(Data >> 32));
}

u64 XRestoration_top_Get_weights(XRestoration_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_WEIGHTS_DATA);
    Data += (u64)XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_WEIGHTS_DATA + 4) << 32;
    return Data;
}

void XRestoration_top_InterruptGlobalEnable(XRestoration_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_GIE, 1);
}

void XRestoration_top_InterruptGlobalDisable(XRestoration_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_GIE, 0);
}

void XRestoration_top_InterruptEnable(XRestoration_top *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_IER);
    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_IER, Register | Mask);
}

void XRestoration_top_InterruptDisable(XRestoration_top *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_IER);
    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_IER, Register & (~Mask));
}

void XRestoration_top_InterruptClear(XRestoration_top *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRestoration_top_WriteReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_ISR, Mask);
}

u32 XRestoration_top_InterruptGetEnabled(XRestoration_top *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_IER);
}

u32 XRestoration_top_InterruptGetStatus(XRestoration_top *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRestoration_top_ReadReg(InstancePtr->Control_BaseAddress, XRESTORATION_TOP_CONTROL_ADDR_ISR);
}

