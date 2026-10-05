// TinyGemmaCpp —— 命令行入口
//
// 输入输出都用 token id：引擎只负责"权重 -> logits -> 下一个 token"，
// 分词交给外部（tools/run.py 用 SentencePiece unigram 做 text<->id）。
// 这样引擎本体零第三方依赖，也方便单独对拍。

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
#include <omp.h>
#endif

#include "config.h"
#include "kv_cache.h"
#include "model.h"
#include "threads.h"

namespace {

void PrintUsage() {
  std::fprintf(stderr,
               "tinygemma —— 从零实现的 Gemma 2 推理引擎 (v0)\n"
               "\n"
               "用法: tinygemma --weights PATH [options]\n"
               "\n"
               "  --weights PATH        .sbs 权重文件（必填）\n"
               "  --prompt-ids LIST     逗号分隔的 prompt token id\n"
               "  --prompt-ids-file F   从文件读 prompt id（任意空白/逗号分隔）\n"
               "  --max-tokens N        最多生成的 token 数（默认 32）\n"
               "  --temperature T       采样温度，0 = 贪心（默认 0）\n"
               "  --top-k K             top-k 采样（默认 40）\n"
               "  --seed S              随机种子（默认 42）\n"
               "  --threads N           线程数（默认物理核数，不是逻辑核）\n"
               "  --dump DIR            导出中间张量到 DIR（.npy）用于对拍\n"
               "  --bench               打印 prefill/decode 吞吐\n"
               "  --profile             打印分阶段耗时（每 token 平均）\n"
               "  --info                只打印模型与权重信息后退出\n");
}

std::vector<int32_t> ParseIds(const std::string& text) {
  std::vector<int32_t> out;
  std::string cur;
  for (size_t i = 0; i <= text.size(); ++i) {
    const char c = (i == text.size()) ? ',' : text[i];
    if (c == ',' || c == ' ' || c == '\n' || c == '\t' || c == '\r') {
      if (!cur.empty()) {
        out.push_back(static_cast<int32_t>(std::strtol(cur.c_str(), nullptr, 10)));
        cur.clear();
      }
    } else {
      cur.push_back(c);
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights_path;
  std::string prompt_ids_text;
  std::string prompt_ids_file;
  std::string dump_dir;
  int64_t max_tokens = 32;
  double temperature = 0.0;
  int64_t top_k = 40;
  uint64_t seed = 42;
  int threads = 0;
  bool bench = false;
  bool info_only = false;
  bool profile = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "参数 %s 缺少取值\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--weights") weights_path = next();
    else if (a == "--prompt-ids") prompt_ids_text = next();
    else if (a == "--prompt-ids-file") prompt_ids_file = next();
    else if (a == "--max-tokens") max_tokens = std::atoll(next().c_str());
    else if (a == "--temperature") temperature = std::atof(next().c_str());
    else if (a == "--top-k") top_k = std::atoll(next().c_str());
    else if (a == "--seed") seed = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--threads") threads = std::atoi(next().c_str());
    else if (a == "--dump") dump_dir = next();
    else if (a == "--bench") bench = true;
    else if (a == "--profile") profile = true;
    else if (a == "--info") info_only = true;
    else if (a == "-h" || a == "--help") { PrintUsage(); return 0; }
    else { std::fprintf(stderr, "未知参数: %s\n", a.c_str()); PrintUsage(); return 2; }
  }

  if (weights_path.empty()) { PrintUsage(); return 2; }

#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
  // 默认不给就用物理核数，别用满逻辑核：超线程在访存受限的 GEMM 里帮不上忙，
  // 反倒让屏障同步和内存争抢拖慢整体（详见 threads.h）。
  if (threads <= 0) {
    threads = tg::DefaultThreads();
    const int phys = tg::DetectPhysicalCores();
    std::fprintf(stderr, "[tinygemma] 线程数自动取 %d（物理核 %d，逻辑核 %u）\n",
                 threads, phys, std::thread::hardware_concurrency());
  }
  omp_set_num_threads(threads);
  omp_set_dynamic(0);  // 不让运行时自己缩减线程数，保持计时可比
#endif

  tg::ModelConfig config = tg::ModelConfig::Gemma2_2B();
  tg::Model model;
  std::string err;
  std::fprintf(stderr, "[tinygemma] 加载权重 %s ...\n", weights_path.c_str());
  const auto t0 = std::chrono::steady_clock::now();
  if (!model.Load(weights_path, config, &err)) {
    std::fprintf(stderr, "[tinygemma] 失败: %s\n", err.c_str());
    return 1;
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double load_s = std::chrono::duration<double>(t1 - t0).count();

  const tg::Weights::Stats& st = model.stats();
  const double mb = 1024.0 * 1024.0;
  const double resident_gb =
      (st.bytes_f32 + st.bytes_bf16 + st.bytes_sfp) / (1024.0 * 1024.0 * 1024.0);
  std::fprintf(stderr,
               "[tinygemma] 就绪  用时 %.2f s\n"
               "  blobs            : %lld\n"
               "  f32 / bf16 / sfp : %.1f MB / %.1f MB / %.1f MB\n"
               "  元素总数         : %.2f M\n"
               "  权重常驻         : %.2f GiB（保持存储宽度，未展开成 f32）\n"
               "  %s\n",
               load_s, static_cast<long long>(st.num_blobs),
               st.bytes_f32 / mb, st.bytes_bf16 / mb, st.bytes_sfp / mb,
               st.elements / 1e6, resident_gb, config.ToString().c_str());

  if (info_only) {
    std::printf("layers=%lld vocab=%lld model_dim=%lld\n",
                (long long)config.num_layers, (long long)config.vocab_size,
                (long long)config.layer.model_dim);
    return 0;
  }

  // ---- prompt ----
  std::vector<int32_t> prompt;
  if (!prompt_ids_file.empty()) {
    std::ifstream in(prompt_ids_file);
    if (!in) { std::fprintf(stderr, "无法读取 %s\n", prompt_ids_file.c_str()); return 1; }
    std::stringstream ss;
    ss << in.rdbuf();
    prompt = ParseIds(ss.str());
  } else if (!prompt_ids_text.empty()) {
    prompt = ParseIds(prompt_ids_text);
  } else {
    std::fprintf(stderr, "[tinygemma] 未提供 prompt，读取 stdin ...\n");
    std::stringstream ss;
    ss << std::cin.rdbuf();
    prompt = ParseIds(ss.str());
  }
  if (prompt.empty()) { std::fprintf(stderr, "prompt 为空\n"); return 1; }

  const int64_t capacity =
      (static_cast<int64_t>(prompt.size()) + max_tokens + 64 + 63) / 64 * 64;
  tg::KVCache kv;
  kv.Init(config.num_layers, capacity, config.layer.kv_heads, config.layer.qkv_dim);
  std::fprintf(stderr, "[tinygemma] KV Cache: %lld 位置 x %lld 层 = %.1f MB\n",
               (long long)capacity, (long long)config.num_layers,
               kv.bytes() / mb);
  if (!dump_dir.empty()) model.SetDumpDir(dump_dir);

  // 计时要在 prefill 之前打开：prefill 和 decode 各记一套，互不干扰。
  if (profile) model.SetProfile(true);

  // ---- prefill ----
  std::vector<float> logits;
  const auto p0 = std::chrono::steady_clock::now();
  model.Forward(prompt, 0, kv, &logits);
  const auto p1 = std::chrono::steady_clock::now();
  const double prefill_s = std::chrono::duration<double>(p1 - p0).count();

  std::vector<int32_t> out;
  uint64_t rng = seed;
  int32_t next = (temperature <= 0.0)
                     ? tg::SampleGreedy(logits.data(), config.vocab_size)
                     : tg::SampleTopK(logits.data(), config.vocab_size,
                                      (float)temperature, top_k, &rng);
  out.push_back(next);

  int64_t pos = static_cast<int64_t>(prompt.size());
  const auto d0 = std::chrono::steady_clock::now();
  while (static_cast<int64_t>(out.size()) < max_tokens) {
    if (next == config.eos_id || next == config.secondary_eos_id) break;
    if (pos >= capacity) { std::fprintf(stderr, "KV Cache 已满\n"); break; }
    model.Forward({next}, pos, kv, &logits);
    ++pos;
    next = (temperature <= 0.0)
               ? tg::SampleGreedy(logits.data(), config.vocab_size)
               : tg::SampleTopK(logits.data(), config.vocab_size,
                                (float)temperature, top_k, &rng);
    out.push_back(next);
  }
  const auto d1 = std::chrono::steady_clock::now();
  const double decode_s = std::chrono::duration<double>(d1 - d0).count();

  if (profile) {
    model.prof_prefill().Report("prefill");
    model.prof_decode().Report("decode");
  }

  if (bench) {
    std::fprintf(stderr,
                 "\n[bench] prefill %lld tok  %.3f s  -> %.2f tok/s (TTFT %.1f ms)\n"
                 "[bench] decode  %lld tok  %.3f s  -> %.2f tok/s (%.1f ms/tok)\n"
                 "[bench] threads = %d\n",
                 (long long)prompt.size(), prefill_s,
                 prompt.size() / (prefill_s > 0 ? prefill_s : 1e-9),
                 prefill_s * 1000.0, (long long)out.size(), decode_s,
                 out.size() / (decode_s > 0 ? decode_s : 1e-9),
                 decode_s * 1000.0 / std::max<size_t>(1, out.size()),
#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
                 omp_get_max_threads()
#else
                 1
#endif
    );
  }

  // 机器可读输出：生成的全部 token id
  for (size_t i = 0; i < out.size(); ++i) {
    std::printf("%s%d", i ? "," : "", out[i]);
  }
  std::printf("\n");
  return 0;
}
