// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XRESTORATION_TOP_H
#define XRESTORATION_TOP_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xrestoration_top_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_BaseAddress;
} XRestoration_top_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XRestoration_top;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XRestoration_top_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XRestoration_top_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XRestoration_top_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XRestoration_top_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XRestoration_top_Initialize(XRestoration_top *InstancePtr, UINTPTR BaseAddress);
XRestoration_top_Config* XRestoration_top_LookupConfig(UINTPTR BaseAddress);
#else
int XRestoration_top_Initialize(XRestoration_top *InstancePtr, u16 DeviceId);
XRestoration_top_Config* XRestoration_top_LookupConfig(u16 DeviceId);
#endif
int XRestoration_top_CfgInitialize(XRestoration_top *InstancePtr, XRestoration_top_Config *ConfigPtr);
#else
int XRestoration_top_Initialize(XRestoration_top *InstancePtr, const char* InstanceName);
int XRestoration_top_Release(XRestoration_top *InstancePtr);
#endif

void XRestoration_top_Start(XRestoration_top *InstancePtr);
u32 XRestoration_top_IsDone(XRestoration_top *InstancePtr);
u32 XRestoration_top_IsIdle(XRestoration_top *InstancePtr);
u32 XRestoration_top_IsReady(XRestoration_top *InstancePtr);
void XRestoration_top_EnableAutoRestart(XRestoration_top *InstancePtr);
void XRestoration_top_DisableAutoRestart(XRestoration_top *InstancePtr);

void XRestoration_top_Set_input_r(XRestoration_top *InstancePtr, u64 Data);
u64 XRestoration_top_Get_input_r(XRestoration_top *InstancePtr);
void XRestoration_top_Set_output_r(XRestoration_top *InstancePtr, u64 Data);
u64 XRestoration_top_Get_output_r(XRestoration_top *InstancePtr);
void XRestoration_top_Set_weights(XRestoration_top *InstancePtr, u64 Data);
u64 XRestoration_top_Get_weights(XRestoration_top *InstancePtr);

void XRestoration_top_InterruptGlobalEnable(XRestoration_top *InstancePtr);
void XRestoration_top_InterruptGlobalDisable(XRestoration_top *InstancePtr);
void XRestoration_top_InterruptEnable(XRestoration_top *InstancePtr, u32 Mask);
void XRestoration_top_InterruptDisable(XRestoration_top *InstancePtr, u32 Mask);
void XRestoration_top_InterruptClear(XRestoration_top *InstancePtr, u32 Mask);
u32 XRestoration_top_InterruptGetEnabled(XRestoration_top *InstancePtr);
u32 XRestoration_top_InterruptGetStatus(XRestoration_top *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
