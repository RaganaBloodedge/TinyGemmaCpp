#include "threads.h"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <cstdio>
#include <cstring>
#endif

namespace tg {
namespace {

#if defined(_WIN32)
// 每个 RelationProcessorCore 记录恰好代表一个物理核（其 GroupMask 里的位
// 是该核上的逻辑处理器）。系统不会把 P 核和 E 核分开报告，这对我们没影响：
// 我们只要"物理核总数"。
int PhysicalCoresWin() {
  DWORD len = 0;
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len) &&
      GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    return 0;
  }
  if (len == 0) return 0;

  std::vector<char> buf(len);
  if (!GetLogicalProcessorInformationEx(
          RelationProcessorCore,
          reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()),
          &len)) {
    return 0;
  }

  // 只能用 off < len 判定：结构体因为带联合体，sizeof 是 80 字节，而
  // RelationProcessorCore 的记录只填 48 字节。拿 sizeof 当上界会漏掉最后
  // 一条记录（在 8P+8E 的机器上正好少算一个 E 核）。
  int cores = 0;
  DWORD off = 0;
  const char* base = buf.data();
  while (off < len) {
    const auto* rec =
        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(base +
                                                                         off);
    if (rec->Size == 0) break;  // 防御：坏数据会死循环
    if (rec->Relationship == RelationProcessorCore) ++cores;
    off += rec->Size;
  }
  return cores;
}
#endif

#if defined(__linux__)
// /proc/cpuinfo 里每个逻辑处理器一段，段内有两行 "physical id" 和 "core id"。
// 同一物理核上的超线程共享这两个值，所以去重后就是物理核数。
//
// 注意容器 / WSL 里这两个字段可能缺失（比如 cgroup 限了 cpu quota），
// 那时直接返回 0 走兜底，不要瞎猜。
int PhysicalCoresLinux() {
  std::FILE* f = std::fopen("/proc/cpuinfo", "r");
  if (f == nullptr) return 0;

  std::set<std::pair<int, int>> seen;
  char line[512];
  int phys = -1;
  int core = -1;

  const auto flush = [&]() {
    if (phys >= 0 && core >= 0) seen.emplace(phys, core);
    phys = -1;
    core = -1;
  };
  const auto value_of = [](const char* s) -> int {
    const char* c = std::strchr(s, ':');
    return c ? std::atoi(c + 1) : 0;
  };

  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::strncmp(line, "physical id", 11) == 0) {
      phys = value_of(line);
    } else if (std::strncmp(line, "core id", 7) == 0) {
      core = value_of(line);
    } else if (line[0] == '\n') {  // 一段结束
      flush();
    }
  }
  flush();
  std::fclose(f);
  return static_cast<int>(seen.size());
}
#endif

}  // namespace

int DetectPhysicalCores() {
#if defined(_WIN32)
  return PhysicalCoresWin();
#elif defined(__linux__)
  return PhysicalCoresLinux();
#else
  return 0;
#endif
}

int DefaultThreads() {
  const int phys = DetectPhysicalCores();
  if (phys > 0) return phys;
  const unsigned hw = std::thread::hardware_concurrency();
  return hw > 0 ? static_cast<int>(std::min<unsigned>(hw, 1024u)) : 1;
}

}  // namespace tg
