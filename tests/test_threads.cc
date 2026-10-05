// 默认并行度探测的不变量测试。
//
// 这类代码没法断言"一定等于 16"——换台机器就变。能断言的只有关系：
// 物理核 ≤ 逻辑核、默认值 ≥ 1、探测成功时默认值就该是物理核数。
// 之前 Windows 侧就因为拿 sizeof 当数组上界漏算了最后一个 E 核
// （15 而非 16），这种错只能靠"物理核 ≤ 逻辑核"这类不变量 + 人工核对发现。

#include <cstdio>
#include <thread>

#include "threads.h"

int main() {
  const int phys = tg::DetectPhysicalCores();
  const int def = tg::DefaultThreads();
  const unsigned hw = std::thread::hardware_concurrency();

  std::printf("逻辑核 (hardware_concurrency) = %u\n", hw);
  std::printf("物理核 (DetectPhysicalCores)  = %d\n", phys);
  std::printf("默认并行度 (DefaultThreads)   = %d\n", def);

  int fail = 0;
  const auto check = [&](bool ok, const char* what) {
    if (!ok) {
      std::printf("  [FAIL] %s\n", what);
      ++fail;
    }
  };

  check(def >= 1, "默认并行度必须 >= 1");
  if (hw > 0) {
    check(static_cast<unsigned>(def) <= hw, "默认并行度不应超过逻辑核数");
    if (phys > 0) {
      check(static_cast<unsigned>(phys) <= hw, "物理核数不应超过逻辑核数");
      check(def == phys, "探测到物理核时，默认并行度应等于物理核数");
    }
  }
  // 逻辑核是 1 的机器（或探测不到）时，物理核只能是 0 或 1
  if (hw == 1 && phys > 1) {
    check(false, "单核机器上不该探测出多个物理核");
  }

  if (fail == 0) {
    std::printf("全部通过\n");
  } else {
    std::printf("失败 %d 项\n", fail);
  }
  return fail ? 1 : 0;
}
