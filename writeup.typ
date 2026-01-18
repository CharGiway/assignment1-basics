#set page(numbering: "1")
#set heading(numbering: "1.")
#set text(font: "Noto Serif CJK SC")
#show heading: set text(font: "Noto Serif CJK SC")

= CS336 Assignment 1 Writeup
作者：蔡志威
日期：2026-01-01

== Problem 1：Understanding Unicode（1 分）
=== （a）
它返回 Unicode 空字符 NULL（U+0000）。

=== （b）
其 `__repr__()` 显示为转义序列 `\x00`，而打印时只是输出一个不可见的控制字符，不显示任何可见字符。

=== （c）
当它出现在文本中时，它作为不可见的嵌入 NUL 字符存在，不会终止或破坏 Python 字符串；连接与打印均成功，但中间没有可见输出。

== Problem 2：Unicode Encodings（3 分）
=== （a）
UTF-8 保留 ASCII 为单字节、对常见语料更紧凑并使字节级 BPE 边界更合理；同时避免 UTF-16/UTF-32 的大小端/BOM 与代理项复杂性，便于流式处理与跨平台一致性。

=== （b）
示例：`b"\xc3\xa9"`（'é' 的 UTF-8 多字节序列）；该函数逐字节解码会把多字节序列拆开，导致 `UnicodeDecodeError` 或错误字符，因此它按字节而非按 UTF-8 序列解码是错误的。

=== （c）
示例：`b"\x80\x80"`；这两个均为续字节且缺少合法的领头字节，属于非法 UTF-8 序列，不能解码为任何 Unicode 字符。

== Problem 3：train_bpe_tinystories（2 分）
=== （a）
在 TinyStories 上训练了 10k 字节级 BPE（包含 `<|endoftext|>`），耗时约 0.14 小时，峰值内存约 11.1 GB；最长 token 为 " accomplishment"（15 字节），为空格前缀的英文长词，合理。

=== （b）
剖析显示并行预分词阶段耗时最多（`_pretokenize_parallel` 累计约 457s），其余时间主要花在进程结果收集与等待上。
== Problem 4：train_bpe_expts_owt（2 分）
=== （a）
在 OpenWebText 上训练了 32k 字节级 BPE 并序列化词表与合并；最长 token 为 64 字节，预览为“ÃÂ”重复 16 次，源自网页语料中的编码 mojibake（UTF‑8 与 Latin‑1 混解），BPE 合并这种高频重复合理但语义上并不合理。

=== （b）
41→TinyStories 的 tokenizer 更干净，最长 token 为“ accomplishment”（15 字节，空格前缀的英文词片），而 OpenWebText 的 tokenizer 合并出“ÃÂ”这类非 ASCII 乱码的超长片段（64 字节），说明 OWT 语料更杂且含编码噪声，字节级 BPE更偏向重复字节模式而非语义词片。
42→

== Problem 5：tokenizer_experiments（4 分）
=== （a）
TinyStories‑10k 的压缩比约为 ≈3.6 bytes/token，OpenWebText‑32k 约为 ≈4.8 bytes/token（更大的词表在通用网页语料上能合出更长片段）。

=== （b）
用 TinyStories‑10k tokenizer 对 OWT 样本，压缩比降至 ≈3.2 bytes/token，HTML/非英文与“mojibake”字节模式更易被拆散、token 更细碎，导致 token 数增加与语料域不匹配。

=== （c）
在 5MB 样本上测得吞吐约 ≈12 MB/s；估算处理 The Pile（825GB 文本）约需 ≈19–20 小时（单机 CPU）。

=== （d）
uint16 可容纳 ≤65,536 的词表，我们的 10k/32k 词表加少量特殊符号均在此范围内；相较 uint32 更省存储与 IO，除非词表 ≥65,536 才需升级到 uint32。

== Problem 6：Transformer LM resource accounting（5 分）
=== （a）
参数总数≈2,127,057,600；单精度加载内存≈8.51 GB（≈7.93 GiB）。

=== （b）
矩阵乘列表（按一次前向，L=1024，d_model=1600，d_ff=6400，num_layers=48，num_heads=25）：
- Q/K/V 投影：3×(L×d_model)·(d_model×d_model)，FLOPs≈6·L·d_model²≈1.572864×10^10/层
- 输出投影 O：1×(L×d_model)·(d_model×d_model)，FLOPs≈2·L·d_model²≈5.24288×10^9/层
- FFN（三次线性，SwiGLU）：(d_model→d_ff 两次，d_ff→d_model 一次)，FLOPs≈6·L·d_model·d_ff≈6.291456×10^10/层
- 注意力内部：QK^T 与 (Attn·V)，FLOPs≈2·H·L²·d_k×2≈6.7108864×10^9/层（d_k=64）
总 FLOPs≈4.51×10^12（48 层汇总加 LM Head，LM Head≈1.646821376×10^11）。

=== （c）
FFN 的三次线性乘占比最高，其次是 Q/K/V 与 O 的线性乘；注意力内部的 QK^T 与 Attn·V 次之，LM Head 占比最小。

=== （d）
取 L=1024、d_ff=4·d_model：
- GPT‑2 small（12 层，d_model=768，H=12）：总≈3.50×10^11；FFN≈49.8%，Q/K/V+O≈16.6%，注意力内部≈11.1%，LM Head≈22.6%
- GPT‑2 medium（24 层，d_model=1024，H=16）：总≈1.03×10^12；FFN≈59.8%，Q/K/V+O≈19.9%，注意力内部≈10.0%，LM Head≈10.2%
- GPT‑2 large（36 层，d_model=1280，H=20）：总≈2.26×10^12；FFN≈64.2%，Q/K/V+O≈21.4%，注意力内部≈8.6%，LM Head≈5.8%
随模型变大，层内线性乘（FFN 与投影，∝d_model² 与 d_model·d_ff）占比上升；注意力内部与 LM Head 的相对占比下降。

=== （e）
将 GPT‑2 XL 的 context_length 提至 16,384：线性乘 FLOPs 随 L 线性增至≈16×，注意力内部随 L² 增至≈256×，总 FLOPs≈1.5×10^14；注意力内部转为主导，FFN 与 LM Head 的相对占比进一步降低。

== Problem 7：learning_rate_tuning（1 分）
在 10 次迭代内：学习率 1e1 的损失很快出现震荡并走高；1e2 与 1e3 几乎立即发散（损失持续增大）。总体上，较大的学习率并未更快衰减损失，而是导致训练发散。
