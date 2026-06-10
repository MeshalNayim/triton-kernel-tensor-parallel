KERNEL_CONFIGS = [
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "num_warps": 8, "num_stages": 4},
    {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 8, "num_stages": 4},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32, "num_warps": 4, "num_stages": 4},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 4},
]


@triton.jit
def matmul_add_relu_kernel_fp16(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_dm,
    stride_dn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k * BLOCK_K
        k_offs = k_start + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn

        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (n_offs[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    c_ptrs = c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    c_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    c = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
    acc += c
    acc = tl.maximum(acc, 0.0)

    d_ptrs = d_ptr + m_offs[:, None] * stride_dm + n_offs[None, :] * stride_dn
    d_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(d_ptrs, acc.to(tl.float16), mask=d_mask)
