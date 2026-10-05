// TinyGemmaCpp —— 默认并行度选择
//
// 为什么不能直接用 std::thread::hardware_concurrency()：
// 它返回的是逻辑核数（含超线程）。在访存受限的 GEMM 里，同一个物理核上的
// 两个超线程抢同一份 L2/L3 和内存带宽，收益接近零，而 OpenMP 的屏障同步
// 开销却按线程数翻倍。实测（i7-14650HX，8P+8E = 16 物理核 / 24 逻辑核）：
//
//   线程数     Windows      WSL
//     12      55.4 ms/tok  54.5 ms/tok
//     16      56.4         54.0      <- 物理核数附近
//     24      59.1         89.9      <- 逻辑核数，Linux 下退化一倍
//
// 所以默认按物理核数走。

#ifndef TINYGEMMA_THREADS_H_
#define TINYGEMMA_THREADS_H_

namespace tg {

// 物理核数量。探测失败（平台不支持或读不到信息）时返回 0。
int DetectPhysicalCores();

// 默认并行度：物理核数优先；探测不到退回 hardware_concurrency()。至少为 1。
int DefaultThreads();

}  // namespace tg

#endif  // TINYGEMMA_THREADS_H_
