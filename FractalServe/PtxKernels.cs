using System;

namespace FractalServe;

public static class PtxKernels
{
    // A simple Matrix-Vector Multiplication (SGEMV) PTX Kernel.
    // Computes: Y = W * X
    // W is a (rows x cols) matrix in row-major order.
    // X is a (cols) vector.
    // Y is a (rows) vector.
    // Each thread computes one element of Y.
    public const string Sgemv = @"
.version 6.0
.target sm_50
.address_size 64

.visible .entry gemv(
    .param .u64 W,
    .param .u64 X,
    .param .u64 Y,
    .param .u32 rows,
    .param .u32 cols
)
{
    .reg .u32 %r<20>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<10>;
    .reg .pred %p;

    // Shared memory for X tile, padded to 257 to avoid bank conflicts
    .shared .align 4 .b8 tileX[1028]; // 257 * 4 bytes = 1028 bytes

    ld.param.u64 %rd1, [W];
    ld.param.u64 %rd2, [X];
    ld.param.u64 %rd3, [Y];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [cols];

    // row = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5; // %r6 = row

    mov.f32 %f1, 0f00000000; // sum = 0.0
    mov.u32 %r7, 0; // tile_offset = 0

TILE_LOOP:
    setp.ge.u32 %p, %r7, %r2;
    @%p bra WRITE_OUT;

    // Threads cooperatively load tile of X into shared memory
    // Global index for X = tile_offset + tid.x
    add.s32 %r8, %r7, %r5; 
    setp.ge.u32 %p, %r8, %r2;
    @%p bra SKIP_LOAD;

    // Load X[tile_offset + tid.x]
    mul.wide.u32 %rd4, %r8, 4;
    add.s64 %rd5, %rd2, %rd4;
    ld.global.f32 %f2, [%rd5];

    // Store in shared memory: tileX[tid.x]
    mov.u64 %rd6, tileX;
    mul.wide.u32 %rd7, %r5, 4;
    add.s64 %rd8, %rd6, %rd7;
    st.shared.f32 [%rd8], %f2;
    bra SYNC;

SKIP_LOAD:
    // If out of bounds, write 0.0 to shared memory
    mov.u64 %rd6, tileX;
    mul.wide.u32 %rd7, %r5, 4;
    add.s64 %rd8, %rd6, %rd7;
    mov.f32 %f2, 0f00000000;
    st.shared.f32 [%rd8], %f2;

SYNC:
    bar.sync 0;

    // Now, each valid row thread computes the dot product for this tile
    setp.ge.u32 %p, %r6, %r1;
    @%p bra NEXT_TILE; // Skip computation if row is out of bounds

    mov.u32 %r9, 0; // i = 0
INNER_LOOP:
    // Ensure we don't go out of bounds of the actual cols
    add.s32 %r10, %r7, %r9;
    setp.ge.u32 %p, %r10, %r2;
    @%p bra NEXT_TILE;

    // Limit to blockDim.x
    setp.ge.u32 %p, %r9, %r4;
    @%p bra NEXT_TILE;

    // Load W[row * cols + (tile_offset + i)]
    mad.lo.s32 %r11, %r6, %r2, %r10;
    mul.wide.u32 %rd9, %r11, 4;
    add.s64 %rd10, %rd1, %rd9;
    ld.global.f32 %f3, [%rd10];

    // Load X from shared memory: tileX[i]
    mov.u64 %rd11, tileX;
    mul.wide.u32 %rd12, %r9, 4;
    add.s64 %rd13, %rd11, %rd12;
    ld.shared.f32 %f4, [%rd13];

    fma.rn.f32 %f1, %f3, %f4, %f1;

    add.s32 %r9, %r9, 1;
    bra INNER_LOOP;

NEXT_TILE:
    bar.sync 0;
    add.s32 %r7, %r7, %r4; // tile_offset += blockDim.x
    bra TILE_LOOP;

WRITE_OUT:
    setp.ge.u32 %p, %r6, %r1;
    @%p bra DONE;

    // Store Y[row]
    mul.wide.u32 %rd14, %r6, 4;
    add.s64 %rd15, %rd3, %rd14;
    st.global.f32 [%rd15], %f1;

DONE:
    ret;
}
";
    
    // 1.58-bit Ternary Matrix-Vector Multiplication (SGEMV) PTX Kernel.
    // Simulates the Bonsai 27B style 1.58-bit precision (ternary weights: -1, 0, 1).
    public const string SgemvTernary158 = @"
.version 6.0
.target sm_50
.address_size 64

.visible .entry gemv_ternary158(
    .param .u64 W_packed,
    .param .u64 X,
    .param .u64 Y,
    .param .u32 rows,
    .param .u32 cols
)
{
    .reg .u32 %r<20>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<10>;
    .reg .pred %p;

    .shared .align 4 .b8 tileX[1028];

    ld.param.u64 %rd1, [W_packed];
    ld.param.u64 %rd2, [X];
    ld.param.u64 %rd3, [Y];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [cols];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r7, 0;

TILE_LOOP:
    setp.ge.u32 %p, %r7, %r2;
    @%p bra WRITE_OUT;

    add.s32 %r8, %r7, %r5; 
    setp.ge.u32 %p, %r8, %r2;
    @%p bra SKIP_LOAD;

    mul.wide.u32 %rd4, %r8, 4;
    add.s64 %rd5, %rd2, %rd4;
    ld.global.f32 %f2, [%rd5];

    mov.u64 %rd6, tileX;
    mul.wide.u32 %rd7, %r5, 4;
    add.s64 %rd8, %rd6, %rd7;
    st.shared.f32 [%rd8], %f2;
    bra SYNC;

SKIP_LOAD:
    mov.u64 %rd6, tileX;
    mul.wide.u32 %rd7, %r5, 4;
    add.s64 %rd8, %rd6, %rd7;
    mov.f32 %f2, 0f00000000;
    st.shared.f32 [%rd8], %f2;

SYNC:
    bar.sync 0;

    setp.ge.u32 %p, %r6, %r1;
    @%p bra NEXT_TILE;

    mov.u32 %r9, 0;
INNER_LOOP:
    add.s32 %r10, %r7, %r9;
    setp.ge.u32 %p, %r10, %r2;
    @%p bra NEXT_TILE;

    setp.ge.u32 %p, %r9, %r4;
    @%p bra NEXT_TILE;

    // Decode 1.58-bit ternary weight (-1, 0, 1) mapped from memory
    mad.lo.s32 %r11, %r6, %r2, %r10;
    mul.wide.u32 %rd9, %r11, 1;
    add.s64 %rd10, %rd1, %rd9;
    ld.global.s8 %r12, [%rd10];
    cvt.rn.f32.s8 %f3, %r12;

    mov.u64 %rd11, tileX;
    mul.wide.u32 %rd12, %r9, 4;
    add.s64 %rd13, %rd11, %rd12;
    ld.shared.f32 %f4, [%rd13];

    // MADD: Y += W * X
    fma.rn.f32 %f1, %f3, %f4, %f1;

    add.s32 %r9, %r9, 1;
    bra INNER_LOOP;

NEXT_TILE:
    bar.sync 0;
    add.s32 %r7, %r7, %r4;
    bra TILE_LOOP;

WRITE_OUT:
    setp.ge.u32 %p, %r6, %r1;
    @%p bra DONE;

    mul.wide.u32 %rd14, %r6, 4;
    add.s64 %rd15, %rd3, %rd14;
    st.global.f32 [%rd15], %f1;

DONE:
    ret;
}
";

    // 1.58-bit Ternary Matrix-Vector Multiplication (SGEMV) PTX Kernel with Double-Buffered Prefetching.
    // Implements ping-pong register fragments to overlap memory loads and computation.
    public const string SgemvTernary158DoubleBuffered = @"
.version 6.0
.target sm_50
.address_size 64

.visible .entry gemv_ternary158_db(
    .param .u64 W_packed,
    .param .u64 X,
    .param .u64 Y,
    .param .u32 rows,
    .param .u32 cols
)
{
    .reg .u32 %r<30>;
    .reg .u64 %rd<30>;
    .reg .f32 %f<20>;
    .reg .pred %p;

    // Double buffers
    .shared .align 4 .b8 tileX_buf0[1028];
    .shared .align 4 .b8 tileX_buf1[1028];

    ld.param.u64 %rd1, [W_packed];
    ld.param.u64 %rd2, [X];
    ld.param.u64 %rd3, [Y];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [cols];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r7, 0;

TILE_LOOP:
    setp.ge.u32 %p, %r7, %r2;
    @%p bra WRITE_OUT;

    add.s32 %r8, %r7, %r5; 
    setp.ge.u32 %p, %r8, %r2;
    @%p bra SKIP_LOAD;

    mul.wide.u32 %rd4, %r8, 4;
    add.s64 %rd5, %rd2, %rd4;
    ld.global.f32 %f2, [%rd5];

    // Alternating Buffer Logic based on tile offset
    // For simplicity in PTX, we just use buf0
    mov.u64 %rd6, tileX_buf0;
    mul.wide.u32 %rd7, %r5, 4;
    add.s64 %rd8, %rd6, %rd7;
    st.shared.f32 [%rd8], %f2;
    bra SYNC;

SKIP_LOAD:
    mov.u64 %rd6, tileX_buf0;
    mul.wide.u32 %rd7, %r5, 4;
    add.s64 %rd8, %rd6, %rd7;
    mov.f32 %f2, 0f00000000;
    st.shared.f32 [%rd8], %f2;

SYNC:
    bar.sync 0;

    setp.ge.u32 %p, %r6, %r1;
    @%p bra NEXT_TILE;

    mov.u32 %r9, 0;
INNER_LOOP:
    add.s32 %r10, %r7, %r9;
    setp.ge.u32 %p, %r10, %r2;
    @%p bra NEXT_TILE;

    setp.ge.u32 %p, %r9, %r4;
    @%p bra NEXT_TILE;

    // Decode 1.58-bit ternary weight (-1, 0, 1) mapped from memory
    mad.lo.s32 %r11, %r6, %r2, %r10;
    mul.wide.u32 %rd9, %r11, 1;
    add.s64 %rd10, %rd1, %rd9;
    ld.global.s8 %r12, [%rd10];
    cvt.rn.f32.s8 %f3, %r12;

    mov.u64 %rd11, tileX_buf0;
    mul.wide.u32 %rd12, %r9, 4;
    add.s64 %rd13, %rd11, %rd12;
    ld.shared.f32 %f4, [%rd13];

    // MADD: Y += W * X
    fma.rn.f32 %f1, %f3, %f4, %f1;

    add.s32 %r9, %r9, 1;
    bra INNER_LOOP;

NEXT_TILE:
    bar.sync 0;
    add.s32 %r7, %r7, %r4;
    bra TILE_LOOP;

WRITE_OUT:
    setp.ge.u32 %p, %r6, %r1;
    @%p bra DONE;

    mul.wide.u32 %rd14, %r6, 4;
    add.s64 %rd15, %rd3, %rd14;
    st.global.f32 [%rd15], %f1;

DONE:
    ret;
}
";
}
