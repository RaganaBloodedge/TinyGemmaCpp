#!/usr/bin/env python3
"""SentencePiece 模型的最小实现（只用 Python 标准库）。

引擎本体只认 token id，text <-> id 的转换全部放在这里。这样引擎零第三方依赖，
而分词逻辑也能单独调试。

.sentencepiece 文件本身是一个 protobuf（sentencepiece_model.proto 的 ModelProto）。
这里只实现需要的字段解析：

    ModelProto
      repeated SentencePiece pieces = 1
      TrainerSpec trainer_spec       = 2
      NormalizerSpec normalizer_spec = 3

    SentencePiece { string piece = 1; float score = 2; Type type = 3; }

    TrainerSpec    { ModelType model_type = 3; int32 vocab_size = 4;
                     bool treat_whitespace_as_suffix = 24; bool byte_fallback = 35; }
    NormalizerSpec { bytes precompiled_charsmap = 2; bool add_dummy_prefix = 3;
                     bool remove_extra_whitespaces = 4; bool escape_whitespaces = 5; }

**注意 model_type**：Gemma 的 tokenizer.spm 是 **BPE**（model_type=2），不是
Unigram。两者切分结果差别很大，用错算法会得到完全不同的 token 序列。BPE 的合并
过程见 `_encode_bpe`，语义与 sentencepiece 的 `bpe::Model::SampleEncode(alpha=0)`
一一对应。

另外，这份 .spm 里的 `score` 并不是常规的对数概率 —— 实测对全部 256000 个 piece
都满足 `score = 473 - id`（id 越小越"常见"，score 越高）。BPE 只把它当合并优先级
用，所以不影响正确性；但如果按 Unigram 去"最大化 Σscore"，结果一定错。

未实现：normalizer 的 `precompiled_charsmap`（Darts 双数组 trie 做的 NFKC 归一化）。
对 ASCII 输入是恒等映射；含全角字符等特殊 Unicode 输入时结果可能与本家不同。
"""

from __future__ import annotations

import heapq
import struct
from dataclasses import dataclass, field

WS_MARKER = "\u2581"  # ▁

# SentencePiece.Type
TYPE_NORMAL = 1
TYPE_UNKNOWN = 2
TYPE_CONTROL = 3
TYPE_USER_DEFINED = 4
TYPE_UNUSED = 5
TYPE_BYTE = 6

# TrainerSpec.ModelType
MODEL_UNIGRAM = 1
MODEL_BPE = 2
MODEL_WORD = 3
MODEL_CHAR = 4

# unigram 里未知符号的惩罚（sentencepiece unigram_model.cc: kUnkPenalty）
_UNK_PENALTY = 10.0


# ---------------------------------------------------------------- protobuf

def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _fields(buf: bytes):
    """按 (field_number, wire_type, payload) 迭代顶层字段。"""
    pos, n = 0, len(buf)
    while pos < n:
        key, pos = _read_varint(buf, pos)
        fnum, wtype = key >> 3, key & 7
        if wtype == 0:  # varint
            val, pos = _read_varint(buf, pos)
            yield fnum, wtype, val
        elif wtype == 2:  # length-delimited
            ln, pos = _read_varint(buf, pos)
            yield fnum, wtype, buf[pos:pos + ln]
            pos += ln
        elif wtype == 5:  # 32-bit
            yield fnum, wtype, buf[pos:pos + 4]
            pos += 4
        elif wtype == 1:  # 64-bit
            yield fnum, wtype, buf[pos:pos + 8]
            pos += 8
        else:
            raise ValueError(f"不支持的 wire type {wtype}")


@dataclass
class Piece:
    piece: str
    score: float
    type: int

    @property
    def is_normal(self) -> bool:
        return self.type in (TYPE_NORMAL, TYPE_USER_DEFINED)


@dataclass
class SentencePieceModel:
    pieces: list[Piece] = field(default_factory=list)
    model_type: int = MODEL_UNIGRAM
    add_dummy_prefix: bool = True
    remove_extra_whitespaces: bool = True
    escape_whitespaces: bool = True
    treat_whitespace_as_suffix: bool = False
    byte_fallback: bool = False
    vocab_size: int = 0

    # 派生索引
    _piece_to_id: dict[str, int] = field(default_factory=dict, repr=False)
    _merge_vocab: dict[str, int] = field(default_factory=dict, repr=False)
    _byte_map: dict[int, int] = field(default_factory=dict, repr=False)
    _ud_by_first: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _unk_id: int = 0
    _max_piece_len: int = 1
    _min_score: float = 0.0

    # ------------------------------------------------------------ 载入
    @classmethod
    def load(cls, path: str) -> "SentencePieceModel":
        with open(path, "rb") as f:
            data = f.read()
        m = cls()
        for fnum, wtype, val in _fields(data):
            if fnum == 1 and wtype == 2:
                m.pieces.append(cls._parse_piece(val))
            elif fnum == 2 and wtype == 2:
                for tf, tw, tv in _fields(val):
                    if tw != 0:
                        continue
                    if tf == 3:        # TrainerSpec.model_type
                        m.model_type = int(tv)
                    elif tf == 4:      # TrainerSpec.vocab_size
                        m.vocab_size = int(tv)
                    elif tf == 24:     # treat_whitespace_as_suffix
                        m.treat_whitespace_as_suffix = bool(tv)
                    elif tf == 35:     # byte_fallback
                        m.byte_fallback = bool(tv)
            elif fnum == 3 and wtype == 2:
                for nf, nw, nv in _fields(val):
                    if nw != 0:
                        continue
                    if nf == 3:
                        m.add_dummy_prefix = bool(nv)
                    elif nf == 4:
                        m.remove_extra_whitespaces = bool(nv)
                    elif nf == 5:
                        m.escape_whitespaces = bool(nv)
        m._build_index()
        return m

    @staticmethod
    def _parse_piece(buf: bytes) -> Piece:
        piece, score, ptype = "", 0.0, TYPE_NORMAL
        for fnum, wtype, val in _fields(buf):
            if fnum == 1 and wtype == 2:
                piece = val.decode("utf-8", "replace")
            elif fnum == 2 and wtype == 5:
                score = struct.unpack("<f", val)[0]
            elif fnum == 3 and wtype == 0:
                ptype = int(val)
        return Piece(piece, score, ptype)

    def _build_index(self) -> None:
        ud: dict[str, list[str]] = {}
        for i, p in enumerate(self.pieces):
            # sentencepiece 的 PieceToId：保留符号优先，之后才是普通 piece。
            # 这里用 setdefault 保持"先出现的胜出"，与遍历顺序一致。
            self._piece_to_id.setdefault(p.piece, i)

            if p.type in (TYPE_NORMAL, TYPE_USER_DEFINED, TYPE_UNUSED):
                # 这三类才进 BPE 的合并词表（sentencepiece InitializePieces）
                self._merge_vocab.setdefault(p.piece, i)
            if p.type == TYPE_USER_DEFINED and p.piece:
                ud.setdefault(p.piece[0], []).append(p.piece)
            if p.type == TYPE_UNKNOWN and self._unk_id == 0:
                self._unk_id = i
            if p.type == TYPE_BYTE and p.piece.startswith("<0x"):
                try:
                    self._byte_map[int(p.piece[3:5], 16)] = i
                except ValueError:
                    pass

        for lst in ud.values():
            lst.sort(key=len, reverse=True)
        self._ud_by_first = ud
        self._max_piece_len = max((len(p.piece) for p in self.pieces), default=1)
        self._min_score = min((p.score for p in self.pieces), default=0.0)

    # ------------------------------------------------------------ 属性
    def __len__(self) -> int:
        return len(self.pieces)

    @property
    def model_type_name(self) -> str:
        return {1: "UNIGRAM", 2: "BPE", 3: "WORD", 4: "CHAR"}.get(self.model_type, "?")

    # ------------------------------------------------------------ 归一化
    def _normalize(self, text: str) -> str:
        if self.escape_whitespaces:
            text = text.replace(" ", WS_MARKER)
        if self.remove_extra_whitespaces:
            while WS_MARKER * 2 in text:
                text = text.replace(WS_MARKER * 2, WS_MARKER)
            text = text.rstrip(WS_MARKER) if self.treat_whitespace_as_suffix \
                else text.strip(WS_MARKER)
        if self.add_dummy_prefix:
            text = text + WS_MARKER if self.treat_whitespace_as_suffix \
                else WS_MARKER + text
        return text

    def encode(self, text: str) -> list[int]:
        norm = self._normalize(text)
        if self.model_type == MODEL_BPE:
            return self._encode_bpe(norm)
        return self._encode_unigram(norm)

    # ------------------------------------------------------------ BPE
    def _user_prefix(self, s: str, i: int) -> str | None:
        """最长 user-defined 符号前缀（sentencepiece 的 PrefixMatcher）。"""
        cands = self._ud_by_first.get(s[i])
        if not cands:
            return None
        for c in cands:
            if s.startswith(c, i):
                return c
        return None

    def _encode_bpe(self, norm: str) -> list[int]:
        """BPE：按合并后的 score 从高到低反复合并相邻符号。

        对应 sentencepiece bpe_model.cc 的 SampleEncode(normalized, alpha=0)：
          - 初始符号 = 一个用户定义符号，或一个 Unicode 字符；它们**不查词表**
          - 只有相邻两符号拼起来的串命中词表时，才作为候选合并
          - 候选按 score 降序处理，score 相同则左端点靠前者优先
          - 用户定义符号标记 freeze，永不参与合并
          - 合并到 UNUSED piece 时记下"反向合并"，最后展开回去
        """
        n = len(norm)
        if n == 0:
            return []

        # ---- 1. 切成初始符号 ----
        sym: list[str] = []
        frozen: list[bool] = []
        i = 0
        while i < n:
            ud = self._user_prefix(norm, i)
            if ud is not None:
                sym.append(ud)
                frozen.append(True)
                i += len(ud)
            else:
                sym.append(norm[i])
                frozen.append(False)
                i += 1
        m = len(sym)
        nxt = [j + 1 if j + 1 < m else -1 for j in range(m)]
        prv = [j - 1 for j in range(m)]

        rev_merge: dict[str, tuple[str, str]] = {}
        # (-score, left, right, size_bytes) —— 等价于 C++ 的 SymbolPairComparator
        heap: list[tuple[float, int, int, int]] = []

        def add_pair(left: int, right: int) -> None:
            if left == -1 or right == -1 or frozen[left] or frozen[right]:
                return
            if not sym[left] or not sym[right]:
                return
            merged = sym[left] + sym[right]
            pid = self._merge_vocab.get(merged)
            if pid is None:
                return
            heapq.heappush(
                heap, (-self.pieces[pid].score, left, right,
                       len(merged.encode("utf-8"))))
            if self.pieces[pid].type == TYPE_UNUSED:
                rev_merge[merged] = (sym[left], sym[right])

        for j in range(1, m):
            add_pair(j - 1, j)

        # ---- 2. 主循环：反复取"最该合并"的一对 ----
        while heap:
            _, left, right, size = heapq.heappop(heap)
            # 过期候选：任一侧已被合并掉，或长度对不上
            if not sym[left] or not sym[right]:
                continue
            if len(sym[left].encode("utf-8")) + len(sym[right].encode("utf-8")) != size:
                continue
            sym[left] = sym[left] + sym[right]
            nxt[left] = nxt[right]
            if nxt[right] != -1:
                prv[nxt[right]] = left
            sym[right] = ""
            add_pair(prv[left], left)
            add_pair(left, nxt[left])

        # ---- 3. 收尾：按链表顺序输出 id ----
        out: list[int] = []
        idx = 0
        while idx != -1:
            if sym[idx]:
                self._emit(sym[idx], rev_merge, out)
            idx = nxt[idx]
        return out

    def _emit(self, w: str, rev_merge: dict[str, tuple[str, str]],
              out: list[int]) -> None:
        pid = self._piece_to_id.get(w, self._unk_id)
        ptype = self.pieces[pid].type if 0 <= pid < len(self.pieces) else TYPE_UNKNOWN
        if ptype == TYPE_UNUSED:
            pair = rev_merge.get(w)
            if pair is None:
                out.append(pid)
                return
            self._emit(pair[0], rev_merge, out)
            self._emit(pair[1], rev_merge, out)
            return
        if ptype == TYPE_UNKNOWN:
            # byte_fallback：未知串按 UTF-8 字节拆成 <0xXX>
            for b in w.encode("utf-8"):
                out.append(self._byte_map.get(b, pid))
            return
        out.append(pid)

    # ------------------------------------------------------------ Unigram
    def _fallback_ids(self, ch: str) -> list[int]:
        out = []
        for b in ch.encode("utf-8"):
            if b in self._byte_map:
                out.append(self._byte_map[b])
        return out

    def _encode_unigram(self, norm: str) -> list[int]:
        """Unigram：最大化 Σ score 的动态规划（本项目的 .spm 用不到，留作对照）。"""
        n = len(norm)
        if n == 0:
            return []
        best = [float("-inf")] * (n + 1)
        back: list[tuple[int, int] | None] = [None] * (n + 1)
        best[0] = 0.0
        for i in range(n):
            if best[i] == float("-inf"):
                continue
            limit = min(self._max_piece_len, n - i)
            matched = False
            for ln in range(limit, 0, -1):
                piece = norm[i:i + ln]
                pid = self._piece_to_id.get(piece)
                if pid is None or not self.pieces[pid].is_normal:
                    continue
                matched = True
                score = best[i] + self.pieces[pid].score
                if score > best[i + ln]:
                    best[i + ln] = score
                    back[i + ln] = (i, pid)
            if not matched:
                best[i + 1] = best[i] + self._min_score - _UNK_PENALTY
                back[i + 1] = (i, -1)
        out: list[int] = []
        pos = n
        while pos > 0:
            entry = back[pos]
            if entry is None:
                break
            i, pid = entry
            if pid == -1:
                out.extend(self._fallback_ids(norm[i:pos]))
            else:
                out.append(pid)
            pos = i
        out.reverse()
        return out

    # ------------------------------------------------------------ 反分词
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """按 sentencepiece 的规则还原文本：byte piece 攒成字节再解 UTF-8。"""
        parts: list[str] = []
        buf = bytearray()

        def flush() -> None:
            if buf:
                parts.append(bytes(buf).decode("utf-8", "replace"))
                buf.clear()

        for i in ids:
            if not (0 <= i < len(self.pieces)):
                continue
            p = self.pieces[i]
            if skip_special and self._is_special(p):
                continue
            if p.type == TYPE_BYTE and p.piece.startswith("<0x"):
                buf.append(int(p.piece[3:5], 16))
                continue
            flush()
            parts.append(p.piece)
        flush()
        text = "".join(parts)
        if self.escape_whitespaces:
            text = text.replace(WS_MARKER, " ")
        return text

    @staticmethod
    def _is_special(p: Piece) -> bool:
        if p.type in (TYPE_CONTROL, TYPE_UNKNOWN, TYPE_UNUSED):
            return True
        # <start_of_turn> / <end_of_turn> 之类是以 user_defined 注册的
        return p.type == TYPE_USER_DEFINED and p.piece.startswith("<") and p.piece.endswith(">")

    def to_pieces(self, ids: list[int]) -> list[str]:
        return [self.pieces[i].piece if 0 <= i < len(self.pieces) else "<?>"
                for i in ids]


if __name__ == "__main__":
    import sys

    m = SentencePieceModel.load(sys.argv[1])
    print(f"pieces      : {len(m)}")
    print(f"model_type  : {m.model_type_name}")
    print(f"vocab_size  : {m.vocab_size}")
    print(f"dummy_prefix: {m.add_dummy_prefix}")
    print(f"escape_ws   : {m.escape_whitespaces}")
    print(f"remove_extra: {m.remove_extra_whitespaces}")
    print(f"byte_fallback: {m.byte_fallback}")
    print(f"max piece   : {m._max_piece_len}")
    print("前 8 个 piece:", [p.piece for p in m.pieces[:8]])
    for probe in ["The capital of France is", "Hello world!", "1 + 1 =", "你好，世界"]:
        ids = m.encode(probe)
        print(f"{probe!r:30} -> {len(ids):>2} tok {ids} -> {m.decode(ids)!r}")
