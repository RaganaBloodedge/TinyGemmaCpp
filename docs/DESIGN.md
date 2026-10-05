# 设计说明

## 数据流总览

```
text ──tools/spm.py──> token ids ──argv/stdin──> main.cc
                                                   │
                                          Model::Forward(tokens, start_pos, kv)
                                                   │
                                        logits[vocab] ──> 采样 ──> next id
                                                           │
                                                    再接回 Forward
```

引擎只认 id。分词放在 Python 侧，好处是引擎本体零第三方依赖、分词逻辑可以单独调试。

## 一次 Forward 的算子顺序

单层（`model.cc:ForwardLayer`），26 层循环，每层 4 个 RMSNorm：

```
x                                  [n, 2304]
│
├─ 1. RMSNorm(pre_att_norm)        → normed
├─ 2. MatMulBT(qkv)                → [n, 4096]
│      行布局: [h0..h7 各 256] ++ [k0,v0,k1,v1,k2,v2,k3,v3 各 256]
├─ 3. RoPE(q 各头, ×query_scale)   ← 原地
│     RoPE(k 各头, ×1.0) 并写 KV Cache
├─ 4. 逐 (token, head)：
│        score[p] = q·k[p]          p ∈ [pos-window+1, pos]
│        soft-cap(50) → softmax → 加权求和 v
├─ 5. att_ein 投影（逐行手写点积，跳过转置）
├─ 6. RMSNorm(post_att_norm) 后加回 x      ← 顺序不能反
│
├─ 7. RMSNorm(pre_ff_norm)         → normed
├─ 8. gating 上半 (gate) / 下半 (up) 两次 MatMulBTRows
├─ 9. GeGLU: c1 = gelu_tanh(c1) * c2
├─ 10. MatMulBT(linear_w)          → [n, 2304]
└─ 11. RMSNorm(post_ff_norm) 后加回 x
```

循环结束后只对**最后一个 token** 做 `final_norm` 与输出投影（`logits = normed_last @ embedᵀ`，
权重绑定），再 soft-cap(30) 得到 logits。

> 注意力为什么按 `(token, head)` 展开：`MatMulBT` 已经是 batched（n 行一次算完），
> 但注意力部分每个 OpenMP 任务只做一次 256 维点积。小序列下并行度够用，
> 代价是没利用矩阵乘的 cache 复用。不过 profile 显示它只占 prefill 的 9.6%
> （见 README「已知限制与下一步」），不是瓶颈。

## 内存布局决策

**权重保持存储宽度，不展开成 f32。** 文件里 63.17% 是 sfp（1 字节/元素），
36.83% 是 bf16（2 字节/元素）。展开成 f32 会让常驻从 2.98 GiB 涨到 ~10 GB，
而且解码本身很便宜（bf16 就是左移 16 位）。代价是 GEMM 内核要为三种类型各写一份。

**GEMM 的行主序不做转置。** HuggingFace 的权重是 `[out, in]`，而 GEMM 习惯
`A[out, in] · B[in, k]`。这里直接实现 `MatMulBT(C, A, B)` 算的是 `A · Bᵀ`，
省掉一次 3 GB 级别的权重转置。

**KV Cache 按位置连续。** `KBase(layer, pos)` 返回 `layer` 层 `pos` 位置的
`kv_heads × head_dim` 连续块，解码时直接 `memcpy` 一行，注意力时不用跳 stride。
128 位置 × 26 层 ≈ 26 MB。

**RoPE 表按需增长。** 启动预热 512 个位置，遇到更长的序列再扩，短对话不浪费内存。

## GEMM 内核与分块

`matmul.h/.cc` 里有两条路：`*Ref` 是标量参考实现（不做任何平台假设，只用来对拍），
默认路径在编译期探到 AVX2/FMA 后走 SIMD。`tests/test_matmul.cc` 用 60 个形状组合
把两者逐用例比对，包括 `k` 不是 32 的倍数、`k < 8`、奇数行数这些边界。

### decode 与 prefill 是两种不同的问题

| | decode（m=1） | prefill（m>1） |
| --- | --- | --- |
| 瓶颈 | 访存带宽 | 计算 / 加载端口 |
| 每个权重元素被用几次 | 1 次 | m 次 |
| 优化目标 | 每字节少花指令 | 提高复用与算术强度 |

所以 `MatMulBTRowsImpl` 对 `m == 1` 和 `m > 1` 走完全不同的循环结构，
`AttEinSum` 也一样（`m == 1` 时反而**不能**做权重解码前移，见 README）。

### prefill：三级分块

```
按 token 分块（MB = 2048KB / (k·4)），块内：
    每个线程拿一个权重行块（并行），块内：
        把一行权重解码一次 → L2/L1 常驻（典型 2304 个 f32 = 9 KB）
        再按 k 分块（KC，默认 1024），每段：
            对 MB 个 token 复用这段权重（权重段只有 KC·4 字节，稳留 L1）
        所有 k 段跑完，把累加缓冲写回 out
```

三个分块各自解决一件事：

- **token 分块**解决激活的重复扫描。用**整块激活的字节数**做预算（2 MB），
  而不是 token 数 —— 因为 `k` 在不同投影间差 4 倍（2304 vs 9216），按 token
  数分会让 `ff linear` 的激活块失控。块的字节数决定权重被重读几遍，
  2 MB 是实测甜点：512KB → 368 GMAC/s，2048KB → 404，3072KB → 376（gate 形状）。
- **每行解码一次**解决解码的重复。解码内联在点积里时同一行被解 `m` 遍，
  每层 54 亿次位运算，和 FMA 抢同一批执行端口。
- **k 分块**解决一级缓存放不下。这是最后一个被找出来的瓶颈，见下。

代价是权重被重读 `m/MB` 遍（几十 MB 量级），远小于收益。

### 微内核：4 token × 2 行

单个点积每 32 个元素要 8 次向量加载换 4 次 FMA，即 **2 loads/FMA**，
而机器只有 2 个 load 端口和 2 个 FMA 端口 —— FMA 只能用一半。
`Gemm4x2Avx2` 让 4 个 a 向量和 2 个 w 向量喂出 8 次 FMA，加载比降到
**0.75 loads/FMA**。

更关键的是 8 个**互相独立**的累加器。FMA 延迟 4 周期，要让两个 FMA 端口都满载
得有 8 条互不依赖的链在飞；2×2 只有 4 条，每周期最多发 4 条，FMA 吞吐卡在一半
（实测正是 43% 峰值）。这一步把 FFN 从 246 → 278 GMAC/s。

微内核的语义是**累加**（`*o += ...`），调用方负责先清零。prefill 路径要靠它做
k 分块累加，`AttEinSum` 要靠它跨 head 累加，两边都需要这个语义。

### 为什么还要在 k 方向分块

4 token 的激活是 `4·k·4` 字节，2 行的权重是 `2·k·4` 字节，k=2304 时分别是
36 KB 和 18 KB，**合计 54 KB，超过 P 核 48 KB 的 L1D**。后果是：同一行对从
一个 token 块扫到下一个 token 块时，权重行已经被激活流挤出 L1，只能回 L2 重读。
而权重行正是最内层唯一的复用点，代价被放大 `n/2` 倍。

验证方法很干净：固定 m 和 n，只改 k 看工作集大小的影响 ——

| k | 工作集 | GMAC/s |
| --- | --- | --- |
| 512 | 12 KB | 464 |
| **1024** | **24 KB** | **535** |
| 2304 | 54 KB | 381 |

峰值正好落在"装得下/装不下"那条线上。于是在行对内部再把 k 切成 1024 一段：
「同一段权重再次被用到」之间只流过 `4·KC·4` = 16 KB 激活，权重段（8 KB）稳留 L1。

代价是输出要多一次累加缓冲中转（每个行对先把整块 token 的结果攒在
`2·m_block` 个 float 里，k 段跑完再写回）。这笔开销远小于省下的 L2 重读，
而且把「按 n 步长散写 out」变成了「连续写 accbuf」。

### 天花板：怎么知道已经到顶了

优化最容易变成无止境的试错。这里先把**天花板量出来**再判断还有多少空间：

1. 用延迟受限的 FMA 依赖链测真实主频（`__asm__` 屏障必须加，否则 GCC 的
   终结值替换会把循环折叠成闭式，测出 100 GHz 这种不可能的读数）：
   单核 4.85 GHz、全核 16 线程 3.33 GHz。
2. 用 8 条独立链测纯 FMA 吞吐（操作数全在寄存器，零访存）：
   单线程 **61 GMAC/s**、16 线程 **602 GMAC/s**。

对照实测：FFN gate 形状单线程 47.2、16 线程 440 GMAC/s —— 也就是
**单线程 77%、16 线程 73%** 的绝对算力上限。而那个上限测试连一次内存访问都没有。

这个比例说明剩下的空间只有约 1.35x，而且都被"必须搬数据"吃掉了，
继续抠微内核的性价比很低。

### sfp 的无分支解码

`dtypes.h` 里的 `SfpDecodeToBf16` 是带 `small/large` 分支的逐元素写法。
`matmul.cc` 里把它化简成闭式：

```
v    = x & 0x7F
bf16 = (v > 63) ? 0x3800 + (v << 4) : 0x3400 + (v << 5)
       | (sign << 15)
```

`tools/verify_sfp_simd.py` 验证 256 个字节里全部 254 个非零编码与分支版逐位相同
（只有 `0x00`/`0x80` 两个零编码需要显式抹零）。

再进一步：解出来的 bf16 只有 16 位，中间量也都装得下 16 位，所以整个解码可以
放在 16 位通道里做 —— 一条向量指令处理的元素数从 8 变成 16。

## `.sbs` 读取器

`io/blob_store.cc` 的算法被重写成约 180 行标量 C++（`sbs_reader.cc`）。不直接 include
原文件的原因是依赖闭包太大：`compression/sfp-inl.h` 重度依赖 Highway SIMD 的
`hn::Vec` / `HWY_IF_U8_D`，需要 foreach_target 多目标编译才有意义；`io/io.h` 又
一路拉出 `util/threading_context.h` 的整套多 socket 线程池 —— 为了复用 ~540 行而
引进半个基础设施，且与本项目自己写的多线程冲突。

移植后用手写的 Python 版本交叉验证：两边独立实现同一份算法，解出的 f32 **位模式**
必须逐位一致（11 组 / 61952 个元素全部通过）。解码是纯位运算，不存在"浮点误差"，
所以这个测试足够强。

读取时用 `std::filesystem::file_size` 而不是 `ftell()` —— Windows 的 `long` 只有
32 位，3.2 GB 的文件会被截断，表现成"文件太小"。

## 分词器

见 README 的"分词器：一个值得记录的坑"。要点：这是 **BPE** 模型，`score` 字段是
合并优先级而非对数概率，必须按 `bpe::Model::SampleEncode(alpha=0)` 的语义实现，
不能按 Unigram 的 Σscore 优化。

验证方式是把官方 sentencepiece 编出来的 `libsentencepiece.a` 链进一个小探针
（`tools/spm_probe.cc`），在同一批文本上逐 id 比对（31 条用例全覆盖）。
