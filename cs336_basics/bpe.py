from __future__ import annotations

import io
from collections import Counter, defaultdict
import json
from typing import Dict, Iterable, List, Tuple
import time
from datetime import datetime, timezone

import regex as re


def _compile_gpt2_pretok_pattern() -> re.Pattern:
    """
    构造 GPT-2 风格的预分词正则表达式。

    规则说明（UTF-8 字节级）：
    - 英文常见缩写后缀：'s, 't, 're, 've, 'm, 'll, 'd
    - 以可选空格开头的字母块或数字块：` ?\p{L}+`、` ?\p{N}+`
    - 其它非空白的符号块：` ?[^ \p{L}\p{N}\s]+`
    - 仅由空白组成且后面不跟非空白的结尾空白：`\s+(?!\S)`（用于捕获结尾换行/空格）
    """
    pat_str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    return re.compile(pat_str)


import cs336_basics.logutil as logutil


def _split_on_special(text: str, special_tokens: List[str]) -> Iterable[str]:
    """
    在特殊标记处切分文本（保持特殊标记原样、避免与邻近文本合并）。
    仅返回普通文本片段，特殊标记不计入频次统计与合并。
    """
    # 如果没有特殊标记，直接返回文本
    if not special_tokens:
        yield text
        return
    pat = re.compile("|".join(re.escape(t) for t in special_tokens))
    pos = 0
    for m in pat.finditer(text):
        if m.start() > pos:
            yield text[pos : m.start()]
        pos = m.end()
    if pos < len(text):
        yield text[pos:]


def _pretokenize(text: str, special_tokens: List[str]) -> Counter[bytes]:
    """
    预分词：
    - 先按特殊标记切分，保证特殊标记不参与 BPE 合并；
    - 对普通片段使用 GPT-2 正则进行分块，并以 UTF-8 编码成字节；
    - 统计每个片段的出现次数，作为初始“词”（字节序列）。
    """
    # 先取出 Pattern，避免重复编译
    pattern = _compile_gpt2_pretok_pattern()
    counts: Counter[bytes] = Counter()
    for segment in _split_on_special(text, special_tokens):
        for m in pattern.finditer(segment):
            s = m.group(0)
            if not s:
                continue
            counts[s.encode("utf-8")] += 1
    return counts


def _get_pair_counts(word_counts: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
    """
    统计所有“词”（字节 ID 序列）内部相邻对的频次，用于选择下一次合并的目标。
    """
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for word, freq in word_counts.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq
    return pair_counts


def _merge_word(word: Tuple[int, ...], a: int, b: int, new_id: int) -> Tuple[int, ...]:
    """
    在给定“词”中，将相邻对 (a,b) 替换为新 ID，返回合并后的“词”。
    """
    out: List[int] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def _pairs(word: Tuple[int, ...]) -> List[Tuple[int, int]]:
    if len(word) < 2:
        return []
    return list(zip(word, word[1:]))


def train_bpe(
    input_path: str | io.BytesIO,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    训练字节级 BPE 分词器。

    参数：
    - input_path：训练语料路径或内存字节流
    - vocab_size：目标词表大小（包含 256 单字节 + 特殊标记 + 合并产生的 token）
    - special_tokens：特殊标记字符串列表（以 UTF-8 加入词表，不参与合并）

    返回：
    - vocab：`{token_id -> token_bytes}` 的映射
    - merges：按创建顺序记录的合并对 `[(bytes1, bytes2), ...]`

    说明：
    - 先用 GPT-2 正则进行预分词，并将片段按 UTF-8 编码为字节序列；
    - 每轮选择最高频的相邻对进行合并；
    - 频次相同时，采用“字典序更大”的 tie-break（按 `(bytes(token1), bytes(token2))` 比较）。
    """
    assert vocab_size > 0
    base_vocab_count = 256 + len(special_tokens)
    # 计算实际需要执行的合并次数，不能超过 vocab_size - base_vocab_count
    num_merges = max(0, vocab_size - base_vocab_count)
    
    text: str
    if isinstance(input_path, io.BytesIO):
        text = input_path.getvalue().decode("utf-8", errors="ignore")
    else:
        with open(input_path, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore")

    logutil.info_kvs(
        "event",  "read_input",
        "input_path",  "BytesIO" if isinstance(input_path, io.BytesIO) else str(input_path),
        "text_len",   len(text),
        "vocab_size",  vocab_size,
        "specials",  len(special_tokens),
        "special_tokens",  json.dumps(special_tokens, ensure_ascii=False),
    )

    # 切成片段并统计出现次数，类似 {b'.': 98136, b',': 55123, b' the': 48886, ...} 的形式
    # 主要就是切分先按照 special token 切分文本，然后对每一个文本执行 pattern 的获取，然后统计一些出现的频次，最终获得一个 dict
    counts = _pretokenize(text, special_tokens)
    sample_counts = dict(counts.most_common(100))
    logutil.info_kvs(
        "event", "pretokenize_summary",
        "unique_pieces", len(counts),
        "counts_sample", sample_counts,
    )
    # 这里最终转为 dict，key 是字节序列，value 是出现的频次
    words: Dict[Tuple[int, ...], int] = {}
    for piece_bytes, freq in counts.items():
        words[tuple(piece_bytes)] = freq

    # 这里产生一个简单的id 到 bytes 的转换
    id_to_bytes: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for t in special_tokens:
        id_to_bytes[len(id_to_bytes)] = t.encode("utf-8")

    # 简单打印一下这个 id_to_bytes 内容，检查是否符合预期
    logutil.info_kvs(
        "event", "id_to_bytes",
        "id_to_bytes", id_to_bytes,
    )

    # 返回结果，需要合并的内容
    merges: List[Tuple[bytes, bytes]] = []

    # pair_counts 就是记录了key=相邻对 value=出现的频次
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    # pair_index 就是记录了 key=相邻对  value=出现的所有词
    pair_index: Dict[Tuple[int, int], set[Tuple[int, ...]]] = defaultdict(set)
    # words 拿到的是一个 dict，key 是字节序列，就是单词，value 是出现的频次
    for w, f in words.items():
        for p in _pairs(w):
            pair_counts[p] += f
            pair_index[p].add(w)
    

    for _ in range(num_merges):
        if not pair_counts:
            break
        # 抽取频次最大的那个
        # 输出一个用于比较的三元组： (kv[1], id_to_bytes[kv[0][0]], id_to_bytes[kv[0][1]])
        # - 第一维：频次 count ，越大越优先
        # - 第二维： id_to_bytes[a] （bytes），用于并列频次时字典序比较
        # - 第三维： id_to_bytes[b] （bytes），用于进一步并列时字典序比较
        # - Python 对元组按字典序比较，先比第一项，再比第二项，以此类推
        best_pair = max(
            pair_counts.items(), # 这里是一个 tuple，key 是相邻对，value 是出现的频次
            key=lambda kv: (kv[1], id_to_bytes[kv[0][0]], id_to_bytes[kv[0][1]]),
        )[0]

        # 添加新的合并和词汇
        a, b = best_pair
        # 新bytes是两个旧 bytes数组的合并
        new_bytes = id_to_bytes[a] + id_to_bytes[b]
        new_id = len(id_to_bytes)
        # 使用新的 ID
        id_to_bytes[new_id] = new_bytes
        # 记录新加入的词汇
        merges.append((id_to_bytes[a], id_to_bytes[b]))

        impacted = pair_index.get(best_pair)
        # 如果已经没有涉及的单词，直接清理一下现场
        if not impacted:
            pair_index.pop(best_pair, None)
            continue

        new_words_acc: Dict[Tuple[int, ...], int] = {}
        for w in list(impacted):
            # 从words 里面，通过 word 来换出现的频次
            f = words.get(w)
            if f is None:
                continue
            old_pairs = _pairs(w)
            for p in old_pairs:
                pc = pair_counts.get(p)
                if pc is not None:
                    pc -= f
                    if pc <= 0:
                        pair_counts.pop(p, None)
                        s = pair_index.get(p)
                        if s is not None:
                            s.discard(w)
                            if not s:
                                pair_index.pop(p, None)
                    else:
                        pair_counts[p] = pc
                        s = pair_index.get(p)
                        if s is not None:
                            s.discard(w)
                            if not s:
                                pair_index.pop(p, None)

            merged = _merge_word(w, a, b, new_id)
            new_pairs = _pairs(merged)
            for p in new_pairs:
                pair_counts[p] = pair_counts.get(p, 0) + f
                s = pair_index.get(p)
                if s is None:
                    pair_index[p] = {merged}
                else:
                    s.add(merged)
            words.pop(w, None)
            new_words_acc[merged] = new_words_acc.get(merged, 0) + f

        for mw, fsum in new_words_acc.items():
            words[mw] = words.get(mw, 0) + fsum
        pair_index.pop(best_pair, None)

    vocab: Dict[int, bytes] = {i: id_to_bytes[i] for i in range(len(id_to_bytes))}
    return vocab, merges
