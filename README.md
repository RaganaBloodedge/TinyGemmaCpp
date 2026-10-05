# TinyGemmaCpp

从零手写的 Gemma 2 推理引擎。C++20，**零第三方依赖**（不链接 gemma.cpp，不用 Highway，
不用 protobuf），直接读 Google 官方发布的 `.sbs` 权重文件并在 CPU 上跑真实模型。

```
$ ./build/tinygemma --weights 2.0-2b-it-sfp.sbs \
      --prompt-ids 2,651,6037,576,6081,603 --max-tokens 14 --bench

[bench] prefill 6 tok  0.141 s  -> 42.70 tok/s (TTFT 140.5 ms)
[bench] decode  14 tok  0.864 s  -> 16.20 tok/s (61.7 ms/tok)
7127,235265,235248,109,2437,736,6218,1382,689,1566,235336,235248,108,107
```

上面这 14 个 token 的前 8 个（`7127,235265,235248,109,2437,736,6218,1382`）与官方
`gemma.cpp` 在同一份权重、同一个 prompt 下的贪心解码结果**逐 token 相同**。

---

## 这是什么

一个完整的、可运行的 Gemma 2 (2B instruction-tuned) 推理实现，包括：

| 部分 | 说明 |
| --- | --- |
| 权重加载 | 自写 `.sbs` 容器解析 + bf16 / sfp(8bit) 解码，约 180 行标量代码，直接读官方权重 |
| 模型前向 | 26 层 transformer：RMSNorm / RoPE / GQA / GeGLU / PostNorm / logit soft-capping |
| KV Cache | 按位置连续存放，支持增量解码 |
| 采样 | 贪心 / top-k + temperature，自带 xorshift64\* 随机数 |
| 并行 | OpenMP 多线程 GEMM，`-march=native` |
| 分词器 | 纯 Python 标准库实现的 SentencePiece（BPE）加载与编解码 |
| 测试 | 算子单元测试 + 两套与外部参考实现的对拍 |

## 这不是什么（诚信边界）

- **不是** gemma.cpp 的复制或改名。没有 include 它的任何头文件，没有链接它的任何库。
  与其构建体系（Highway SIMD、多 socket 线程池、sentencepiece、protobuf）完全无关。
- 从 gemma.cpp 移植的只有**算法层面的知识**：`.sbs` 容器布局、sfp 位运算解码、
  Gemma 2 的算子顺序与超参。这些都在对应文件头部注明了出处（Apache-2.0）。
- 引擎是通用工具，权重是测试对象 —— 本仓库不带权重。

## 正确性验证

推理引擎最容易"看起来能跑但其实是错的"。下面三层是**与外部参考实现**的交叉比对，
第四层是算子级单元测试：

### 1. 权重解码：与独立 Python 实现逐位比对

C++ 和 Python 各写一遍 `.sbs` 解析 + sfp/bf16 解码。解码是纯位运算，不存在"浮点
误差"，任何一位不同都是 bug。

```
$ python tools/verify_decode.py --sbs 2.0-2b-it-sfp.sbs --bin build/test_sbs.exe
  [ ok ] c_final_norm     off=0            n=2304   逐位一致
  [ ok ] c_embedding      off=0            n=4096   逐位一致
  ... (11 组，覆盖三种存储类型、含中间偏移)
全部通过：11 组 / 61952 个元素，位模式完全一致
```

### 2. 分词器：与官方 SentencePiece 库逐 id 比对

`tools/spm_probe.cc` 链接 sentencepiece 构建出的 `libsentencepiece.a`，把同一批文本
两边各切一次。探针跑在 WSL 里，路径随机器而变，用环境变量给：

```
$ TG_SPM_PROBE=/path/in/wsl/spm_probe \
  TG_MODEL_IN_WSL=/path/in/wsl/tokenizer.spm \
  python tools/verify_tokenizer.py
  [ ok ] 'The capital of France is'                         5 tok
  [ ok ] 'The quick brown fox jumps over the lazy dog.'    10 tok
  [ ok ] '你好，世界'                                            3 tok
  [ ok ] 'Hello 👋 world 🌍'                                  5 tok
  ... (31 条：英文 / 代码 / 数字 / 空白 / 中日韩俄 / emoji / 长文本)
全部通过：31 条用例与官方 sentencepiece 完全一致
```

> 在 Windows 的 Git Bash 下设这两个变量时要带 `MSYS_NO_PATHCONV=1`：
> MSYS 会把看起来像 Unix 路径的值（`/home/...`、`/mnt/...`）改写成 Windows 路径，
> 传给 WSL 就成了不存在的文件。这个失败现象是"探针无输出"，很容易误判成探针坏了。

### 3. 端到端：与官方 gemma.cpp 逐 token 比对

同一份权重、同一个 prompt、同样贪心：

| | 输出 |
| --- | --- |
| 官方 `gemma.cpp` | `Paris. \n\nIs this statement true` |
| TinyGemmaCpp | ` Paris. \n\nIs this statement true` |

（前导空格来自 sentencepiece 的反分词规则，两边一致；官方打印时裁掉了它。）

把这句输出文本送回我们的分词器，得到的 id 序列是
`[7127, 235265, 235248, 109, 2437, 736, 6218, 1382]` —— 与引擎自己生成的 8 个 token
**完全相同**。这说明从权重解码、26 层前向、KV Cache 到采样，整条链路的数值都是对的。

### 4. 算子单元测试

```
$ ./build/test_ops
  [ ok ] RoPE 配对是 (0, 4) 而不是相邻的 (0, 1)
  [ ok ] gelu_tanh(1) = 0.841192（不是 erf 版 0.841345）
  [ ok ] cap 把极端 logit 压到 30 以内
  [ ok ] 概率和为 1 (1)
全部通过
```

## 性能

机器：i7-14650HX（8 P-core + 8 E-core，24 逻辑核，只有 AVX2/FMA，无 AVX-512），
16 线程（= 物理核数）。移动平台的绝对吞吐受功耗预算与后台负载影响，**比值比绝对值更有参考价值**。

内存：权重常驻 **2.98 GiB**。引擎保持文件里的存储宽度（36.83% bf16 + 63.17% sfp），
不展开成 f32 —— 展开要 ~10 GB。

### 两轮 GEMM 优化

**第一轮**（解决"解码被重复"和"激活被重复扫"）：

| 场景 | 优化前 | 优化后 | 加速 |
| --- | --- | --- | --- |
| prefill 256 token（TTFT） | 9038 ms (28.3 tok/s) | **2080 ms (123.1 tok/s)** | **4.3x** |
| prefill 768 token（TTFT） | 31769 ms (24.2 tok/s) | **7811 ms (98.3 tok/s)** | **4.1x** |
| decode（短上下文） | ~70 ms/tok | ~58 ms/tok | 1.2x |

**第二轮**（解决"一级缓存放不下"）。同一台机器、同一份编译产物，只切环境变量
把旧参数还原出来做交替 A/B，四轮方差约 2%：

| 场景 | 旧参数 | 新参数 | 加速 |
| --- | --- | --- | --- |
| prefill 301 token | 2.562 s | **2.083 s** | **1.23x** |
| prefill 768 token | 8.021 s | **6.668 s** | **1.20x** |

内核级（`tests/bench_gemm_batch.cc`，m=301，16 线程，GMAC/s；多次测量取代表值）：

| 形状 | 第二轮前 | 第二轮后 |
| --- | --- | --- |
| ff gate/up (9216×2304) | 368 | **~420** |
| ff linear (2304×9216) | 235 | **~353** |

`ff linear` 的 k 是 9216、m 只有 301，激活块按 token 数分会严重超预算，
改成按字节数给预算后收益最大（+50%）。

分阶段剖析（301 token 的 prefill，每 token 毫秒，`--profile`）：

| 阶段 | 两轮前 | 第一轮后 | 第二轮后 |
| --- | --- | --- | --- |
| ff gate/up | 17.76 | 3.46 | **2.92** |
| ff down | 9.09 | 2.73 | **1.70** |
| qkv | 3.94 | 0.78 | 0.66 |
| attention | 0.75 | 0.69 | 0.65 |
| att_ein | 2.28 | 0.43 | 0.42 |
| geglu | 1.67 | 0.14 | 0.13 |
| **合计** | **35.87** | **8.54** | **6.75** |

### 与官方 gemma.cpp 对比

同机、同权重、同线程数（16），两边都用 WSL 里的 GCC 13.3 编译
（`tools/compare_official.sh`）：

| prompt | 指标 | TinyGemmaCpp | 官方 gemma.cpp |
| --- | --- | --- | --- |
| 6 token | TTFT | **120 ms** | 178 ms |
| | decode | **21.4 tok/s** | 15.6 tok/s |
| 301 token | prefill | 2.69 s (112 tok/s) | **2.24 s (134 tok/s)** |

短 prompt 的 TTFT 与 decode 都领先；长 prompt 的 prefill 落后约 1.20x（第二轮前是 1.30x）。

> **编译器差异会影响这个结论**：同一份源码，Windows（MSYS2 GCC 16.1）
> 编译出的 prefill 301 token 只要 2.08 s，比 WSL（GCC 13.3）快约 30%。
> 上表特意两边都用 WSL 的编译器，是为了跟官方引擎站在同一起跑线上。

### 做过哪些优化

每项改动前都先看 profile，按数据决定改哪里：

1. **权重按行解码一次**（第一轮 4.3x 里的大头）。原始写法把解码内联在点积里，
   同一行权重被解了 `m` 遍：256 token × 9216 行 × 2304 维 = 每层 54 亿次位运算。
   改成"按 token 分块 → 块内解码一行到缓存 → 对块内所有 token 复用"。
2. **token 分块**。不分块时每读一行权重都要把全部 token 的激活重扫一遍，
   激活流量 = `n·m·k·4` 字节，每层 21.7 GB、26 层 565 GB。
   **这一步单独做只快 1.6%**——因为解码重复的代价更大，两者必须一起改才见效。
3. **AVX2 寄存器分块微内核**：`4 token × 2 行`。8 个独立累加器把 FMA 的 4 周期
   延迟完全藏住（只有 4 条链时每周期最多发 4 条，FMA 端口只用一半），
   同时加载比从 2 loads/FMA 降到 0.75。
4. **att_ein（o_proj）解码前移**（第一轮 prefill 段 5.4x）：同一个毛病——解码被重复 `m` 遍。
   按 out_dim 分块、每块把所有 head 的权重解码一次。
   decode（m=1）**不能**这么改：每个权重只用一次，多绕一趟"解码成 f32 再点乘"
   反而比融合解码的点积慢，所以 `m=1` 单独保留融合路径。
5. **向量化 GeGLU**（7x）：化简 `gelu_tanh(v) = 0.5·v·(1+tanh(arg)) = v·σ(2·arg)`，
   只用一次 exp + 一次除法，把逐元素的 libm `std::tanh` 调用换掉。
   对双精度参考，v ∈ [-20, 20] 上最大绝对误差 **4.2e-07**（`test_ops` 里有这条）。
6. **默认线程数取物理核而非逻辑核**：这台机器是 8P+8E，把线程数开到 24（逻辑核）
   在 WSL 下 decode 会退化到 89.9–183.6 ms/tok，而 16 物理核是 54 ms。
   `src/threads.cc` 用 `GetLogicalProcessorInformationEx`（Windows）和
   `/proc/cpuinfo` 的 `(physical id, core id)`（Linux）探测物理核数。
7. **token 分块改成按字节给预算**（第二轮）。原来按固定 512 KB 激活算块大小，
   隐式假设了 k 是 2304；`ff linear` 的 k 是 9216，激活块只有 1/4 大、权重被重读
   22 遍。改成 2 MB 字节预算后 `ff linear` +39%。再大反而变慢（激活块挤爆 L3）。
8. **k 方向分块**（第二轮，+14–17%）。4 token 的激活 + 2 行的权重在 k=2304 时
   是 36 + 18 = 54 KB，**超过 P 核 48 KB 的 L1D**，导致权重行在 token 扫描过程中
   被挤出去、每次回 L2 重读。按 k 切成 1024 一段后工作集降到 24 KB，稳留 L1。
   验证方法与推导见 `docs/DESIGN.md`。


## 构建与运行

需要 C++20 编译器（g++ / clang++）+ OpenMP。没有其他依赖。

```bash
./build.sh              # 产出 build/tinygemma
./build.sh --tests      # 额外产出四个测试程序
./build.sh --bench      # 额外产出两个内核微基准（不需要权重）
```

### 先准备权重

本仓库不含权重：模型文件由 Google 单独发布，转成 `.sbs` 的工具在 gemma.cpp 里。
准备好之后放哪个目录都行——脚本按 `TG_WEIGHTS` / `TG_TOKENIZER` 环境变量 →
`<仓库>/weights/` → 同级 `gemma.cpp-main/weights/` 的顺序查找，不必改代码：

```bash
mkdir -p weights
cp /path/to/2.0-2b-it-sfp.sbs weights/
cp /path/to/tokenizer.spm     weights/
```

引擎本身只认 token id，文本转换在 Python 侧：

```bash
# 对话模式（套 Gemma 2 instruction 模板）
python3 tools/run.py --question "What is the capital of France?" --threads 16 --bench

# 原始续写
python3 tools/run.py --prompt "Once upon a time" --style raw --max-tokens 40 --show-ids
```

## 目录结构

```
src/
  dtypes.h        bf16 / sfp 的存储类型与解码（sfp 是 15 行位运算）
  config.h/.cc    Gemma 2 2B 的超参（含滑动窗口的隔层交替规则）
  sbs_reader.h    .sbs 容器解析：header + 目录 + 256B 对齐负载
  weights.h/.cc   把 blob 按 key 映射成带形状的张量视图
  matmul.h/.cc    GEMM 内核：标量参考 + AVX2（含 4x2 寄存器分块微内核）
  ops.h/.cc       RMSNorm / RoPE / GeGLU / softmax / soft-cap（含 AVX2 版 exp/tanh/sigmoid）
  kv_cache.h      KV Cache（按位置连续）
  model.h/.cc     26 层前向 + 采样 + 分阶段计时
  threads.h/.cc   物理核探测，决定默认并行度
  main.cc         CLI

tests/
  test_ops.cc      算子数值测试（含向量化 gelu 的精度扫描）
  test_matmul.cc   GEMM 内核回归：SIMD 路径 vs 标量参考，60 个形状组合
  test_sbs.cc      .sbs 解析 + 张量形状校验 + hex dump（供对拍）
  test_threads.cc  物理核探测的不变量
  bench_kernels.cc 内核微基准，m=1（decode）形状，不需要权重文件
  bench_gemm_batch.cc 内核微基准，m>1（prefill）形状 + 纯 FMA 天花板测量

tools/
  paths.py        权重 / 分词器的路径发现（环境变量 -> 仓库内 -> 同级 gemma.cpp-main）
  spm.py          纯标准库 SentencePiece（BPE）加载 / 编码 / 解码
  run.py          端到端驱动：text -> ids -> 引擎 -> ids -> text
  sbs.py          .sbs 的 Python 参考实现（对拍用）
  verify_decode.py    C++ vs Python 解码逐位对拍
  verify_tokenizer.py 我们的分词器 vs 官方 sentencepiece
  verify_sfp_simd.py  验证 sfp 的无分支闭式解码与逐元素分支版等价
  spm_probe.cc        调官方 sentencepiece 的探针（需自行链接 libsentencepiece）
  bench_prefill.py    按 prompt 长度扫 TTFT，每档取最快一次
  compare_official.sh 同机同权重与官方 gemma.cpp 正面对比
  scan_threads.sh     线程数扫描（每个配置多次取最快值）
  diag_omp.sh         Linux 侧 OpenMP 亲和/自旋策略的影响
  diag_wsl.sh         线程数与 OMP_PROC_BIND 的交叉扫描
  diag_wsl2.sh        只改环境变量，定位多线程退化的来源
```

CLI 常用开关：

```bash
--profile     # 打印分阶段耗时，prefill 与 decode 分开记账
--threads N   # 默认取物理核数，不是逻辑核
--bench       # 打印 prefill/decode 吞吐
```

调参用的环境变量（正常跑不需要设，见 `src/matmul.cc` 顶部注释）：

```bash
TG_MBLOCK_KB=2048   # 每块激活的 KB，决定权重被重读几遍
TG_KC=1024          # k 方向分块宽度，决定微内核工作集能否留在 L1
TG_USCHED=4         # OpenMP 行对循环的调度块
```

## 实现要点

### Gemma 2 的坑（都踩过并写进了测试）

| 项 | 容易写错的地方 | 正确做法 |
| --- | --- | --- |
| RMSNorm | 用 `w` 做缩放 | Gemma 2 用 `(1 + w)`，源码里就是 `MulAdd(m, vw, m)` |
| RoPE | 相邻配对 `(2i, 2i+1)` | Gemma 用 `(i, i + d/2)` |
| 注意力 | 没有 soft-cap | softmax **之前**做 `50 · tanh(x/50)` |
| 输出 logits | 没有 soft-cap | 采样**之前**做 `30 · tanh(x/30)` |
| 残差 | `x = x + branch(norm(x))` | 先对分支输出做 norm，再加回主干 |
| 归一化层数 | 每层 2 个 | 每层 **4** 个（attn 前后 + FFN 前后） |
| embedding | 不缩放 | 乘 `sqrt(2304) = 48` |
| query scale | 忘了乘 | `1/16`（即 `head_dim^-0.5`），RoPE 时一起乘 |
| GeGLU | 用 erf 版 gelu | Gemma 用 tanh 近似 |
| QK-norm | 以为 v0 要做 | **Gemma 2 没有**，那是 Gemma 3 |

### 权重格式：`.sbs`

```
Header  16B : magic(u32)="SBS\n" | num_blobs(u32) | file_bytes(u64)
Directory   : 每条 32B = key(16B, ASCII 明文, \0 填充) + offset(u64) + bytes(u64)
Payload     : 每个 blob 256B 对齐
```

key 的首字符就是存储类型：`F`=f32、`B`=bf16、`$`=sfp、`2`=NUQ(4bit)、`I`=i8。
这份 2B 权重是 pre-2025 格式，**不含** config 与 tokenizer，两者都要外部给。

`sfp` 是只有 1 字节的浮点：bit7 符号，bit6 决定指数字段长度，指数 4 bit，
尾数 2 或 3 bit。可表示范围约 `[1.19e-7, 1.875]`，所以 sfp 张量的 `|x|` 必须 ≤ 1.875
（本模型的 sfp 张量 scale 全是 1.0，不需要额外缩放）。

### 分词器：一个值得记录的坑

`tokenizer.spm` 是 **BPE** 模型（`trainer_spec.model_type = 2`），不是 Unigram。
更反直觉的是它的 `score` 字段不是对数概率 —— 实测全部 256000 个 piece 都满足
`score = 473 - id`。

第一版按"Unigram 最大化 Σscore"实现，`The capital of France is` 被切成 11 个 token
（`▁T|he|▁c|ap|it|al|▁of|▁F|ran|ce|▁is`），而正确切分是 5 个
（`The|▁capital|▁of|▁France|▁is`）。因为 BPE 只把 score 当**合并优先级**用，
按 Σscore 去优化它完全没有意义。

修正后的实现严格对齐 `bpe::Model::SampleEncode(alpha=0)`：初始符号不查词表、
相邻对按 score 降序合并、score 相同则左端点靠前者优先、user-defined 符号 freeze、
合并到 UNUSED piece 时记录反向合并以便最后展开。

模型参数：`add_dummy_prefix=0`、`remove_extra_whitespaces=0`、`escape_whitespaces=1`、
`byte_fallback=1`。未实现 `precompiled_charsmap` 的 NFKC 归一化 —— 对 ASCII 是恒等映射。

## 已知限制与下一步

GEMM 已接近这台机器的算力上限，继续抠微内核的收益有限。

量天花板的办法（`build/bench_gemm_batch --fma`）：用 8 条独立 FMA 链测纯吞吐，
操作数全在寄存器、零访存。结果是**单线程 61 GMAC/s、16 线程 602 GMAC/s**。
对照 FFN gate 形状的实测（单线程 47.2、16 线程 440），即 **77% / 73%**。
剩下那约 1.35x 的空间全部要拿来搬运数据（激活、权重、解码），拿不满。

按优化价值排序，剩下的都是"减少搬运量"而不是"改内核"：

- **长 prompt 的 prefill 仍比官方 gemma.cpp 慢 1.20x**（同编译器、同线程数）。
- **decode 已经贴到访存天花板**：每 token 要搬 3.08 GB 权重，纯读带宽上限
  69.9 GB/s 对应 22.7 tok/s，实测 16–21 tok/s。再往上只能减少搬运量。
- **`logits` 投影在 decode 里占 24%**：词表 256000 × 2304 是 bf16（1125 MB/token），
  已经跑在带宽上限。除非把 embedding 也压成 8bit，否则没空间。
- **attention 仍然逐 `(token, head)` 做点积**，没合成 `Q·Kᵀ` 大矩阵乘。
  它现在只占 prefill 的 9.6%，不是当前瓶颈。
- **4bit (NUQ) 权重**是唯一能同时改善 decode（少搬一半权重）和 prefill 的方向，
  代价是新的解码路径与一套新的精度验证。
- 批处理 / continuous batching、CUDA 后端均未实现。
- 滑动窗口已按配置实现（偶数层 4096 / 奇数层 8192），序列 ≤ 4096 时与全局注意力等价，
  长序列路径尚未做过专门测试。

### 测量方法上的四个坑

- **必须用"最快的一次"而不是平均值**：机器上常有别的负载（实测同一配置的
  decode 会在 58–84 ms/tok 之间跳），平均值会把干扰算进来。
  `tools/scan_threads.sh` 每个配置跑 N 次取最快。
- **对比时要注意 page cache 冷启动**：权重在 3 GB 量级，先跑的进程要承担
  首次缺页的开销（WSL 下从 `/mnt/<盘符>` 读 Windows 分区尤其明显，冷启动多花 ~10 s）。
  两边对比前先 `cat` 一遍权重把缓存预热。
- **测主频 / 算力上限时，循环必须加汇编屏障**：`x = fma(x, y, z)` 这种写法会被
  GCC 的终结值替换折叠成闭式，实测能报出 100 GHz、1400 GMAC/s 这种物理上
  不可能的数字。加 `__asm__ __volatile__("" : "+x"(x))` 之后读数才可信——
  本项目第一次量"纯 FMA 峰值"就是这么翻的车。
- **"改了一点"要用交替 A/B 而不是前后各测一遍**：这台机器的背景负载在分钟级
  尺度上会漂移，前后各跑一遍很容易把噪声当成收益（第二轮优化里前测 2.57 s、
  后测 2.00 s 看着像 1.29x，交替 A/B 出来才能确定是 1.23x）。现在三个分块参数
  都能用环境变量还原旧行为，正好用来做这种交替对照。
