// 用官方 sentencepiece 库给同一份 tokenizer.spm 分词。
// 用途：回答"我们手写的 Python 分词器切得对不对"。
//
//   spm_probe MODEL.spm 文本...        逐条分词
//   spm_probe MODEL.spm < lines.txt    每行一条（避开命令行转义问题）
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "builtin_pb/sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"

namespace {

void Show(sentencepiece::SentencePieceProcessor& spp, const std::string& in) {
  std::vector<int> ids;
  std::vector<std::string> pieces;
  spp.Encode(in, &ids);
  spp.Encode(in, &pieces);
  std::printf("IN   %s\n", in.c_str());
  std::printf("IDS  (%zu):", ids.size());
  for (int id : ids) std::printf(" %d", id);
  std::printf("\nPC   :");
  for (const auto& p : pieces) std::printf(" [%s]", p.c_str());
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s MODEL.spm [text...]\n", argv[0]);
    return 2;
  }
  sentencepiece::SentencePieceProcessor spp;
  const auto st = spp.Load(argv[1]);
  if (!st.ok()) {
    std::fprintf(stderr, "load failed: %s\n", st.ToString().c_str());
    return 1;
  }
  const auto& ns = spp.model_proto().normalizer_spec();
  const auto& ts = spp.model_proto().trainer_spec();
  std::printf("vocab=%d\n", spp.GetPieceSize());
  std::printf("model_type=%d (1=UNIGRAM 2=BPE 3=WORD 4=CHAR)\n",
              static_cast<int>(ts.model_type()));
  std::printf("byte_fallback=%d treat_ws_as_suffix=%d vocab_size=%d\n",
              static_cast<int>(ts.byte_fallback()),
              static_cast<int>(ts.treat_whitespace_as_suffix()),
              ts.vocab_size());
  std::printf("add_dummy_prefix=%d remove_extra_ws=%d escape_ws=%d\n",
              static_cast<int>(ns.add_dummy_prefix()),
              static_cast<int>(ns.remove_extra_whitespaces()),
              static_cast<int>(ns.escape_whitespaces()));

  if (argc > 2) {
    for (int i = 2; i < argc; ++i) {
      if (std::string(argv[i]) == "-") {
        // 从 stdin 读整段文本作为一条输入（避开命令行编码问题）。
        // 不剥末尾换行 —— 换行本身也是要分词的字符，由调用方决定给什么。
        std::string all((std::istreambuf_iterator<char>(std::cin)),
                        std::istreambuf_iterator<char>());
        Show(spp, all);
      } else {
        Show(spp, argv[i]);
      }
    }
    return 0;
  }
  std::string line;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    Show(spp, line);
  }
  return 0;
}

