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

== Problem 8：adamwAccounting（2 分）
=== （a）
设批量为 `B=batch_size`，上下文长度 `T=context_length`，层数 `L=num_layers`，模型维度 `d=d_model`，头数 `h=num_heads`，词表大小 `v=vocab_size`，并令 `d_ff=4d`、`d_k=d/h`。采用 float32（每元素 4 bytes）。
- 参数：`P = 2vd + L(4d^2 + 3d d_ff + 2d) + d`；参数内存：`M_param = 4P`
- 梯度：`M_grad = 4P`
- 优化器状态（AdamW 两个动量）：`M_opt = 8P`
- 激活（仅计题目指定组件）：
  - 每层：`B[16 T d + 2 h T^2]`（2×RMSNorm + QKV + 加权和 + 输出投影共计 `8Td`；FFN 的 W1、SiLU、W2 共计 `8Td`；注意力内 `Q^T K` 与 `Attn·V` 共计 `2hT^2`）
  - 末层与输出：`B(T d + T v + T)`（final RMSNorm、输出嵌入 logits、交叉熵标量）
  - 激活内存：`M_act = 4[ L·B(16Td + 2hT^2) + B(Td + Tv + T) ]`
- 总峰值：`M_total(B) = M_param + M_grad + M_opt + M_act`

=== （b）
GPT‑2 XL 设定：`v=50,257`，`T=1,024`，`L=48`，`d=1,600`，`h=25`，`d_ff=6,400`。
- `P = 2,127,057,600`；`M_param + M_grad + M_opt = 16P ≈ 34.03 GB`
- 每样本激活：`a = 4[ L(16Td + 2hT^2) + Td + Tv + T ] ≈ 15.31 GB`
- 故总内存：`M_total(B) ≈ 15.31 · B + 34.03 (GB)`；在 80 GB 预算下最大 `B = ⌊(80−34.03)/15.31⌋ = 3`。

=== （c）
一次 AdamW 步的 FLOPs（忽略 softmax 常数项与规范化小项，计主要矩阵乘；更新开销 `O(P)`）：
- 前向每层：`8BTd^2 + 6BTd d_ff + 4BT^2 d`；LM Head：`2BTdv`
- 前向总计：`F_fwd = L(8BTd^2 + 6BTd d_ff + 4BT^2 d) + 2BTdv`
- 反向约为前向 2 倍；一步总 FLOPs：`F_step ≈ 3 · F_fwd + O(P)`

=== （d）
单卡 A100（FP32 峰值 `19.5 TFLOP/s`），`MFU=50%` 则有效吞吐 `9.75 TFLOP/s`。以 GPT‑2 XL、`B=1024`、`T=1024`、`L=48`：
- `F_fwd ≈ 4.513×10^12 · B` → `F_step ≈ 3 · 4.513×10^12 · B ≈ 1.386×10^16 FLOPs/步`
- 训练 `400k` 步耗时：`400,000 · 1.386×10^16 / 9.75×10^12 ≈ 5.69×10^8 s ≈ 6,580 天`
（按前向×3 的步开销与 50% MFU 估算）

== Problem (learning_rate): Tune the learning rate（3 分）
=== 设定与策略
我们在 TinyStories‑10k 上用 CS336 的小模型设定进行学习率调优：`vocab_size=10000`、`context_length=256`、`d_model=512`、`d_ff=1344`、`num_layers=4`、`num_heads=16`、`batch_size=32`，总步数 5000（40,960,000 tokens）。学习率采用“线性 warmup → 余弦退火”并在第 `X` 步精确到达 `min_lr`（余弦调度见 `cs336_basics/optim/lr_schedule.py:11-16`，训练循环见 `cs336_basics/train_lm.py:102-107`）。搜索策略：先在基线 `max_lr=1e-4` 附近做小范围网格（`1e-4→1.2e-4`），再在更高学习率上做短试跑以观察稳定性，必要时配合 `beta2/weight_decay` 微调。

=== 学习率扫参结果（MPS，低资源）
- 运行 A（高学习率设定）：`max_lr=6e-4, min_lr=3e-4, warmup=2000, cosine_cycle_iters=5000, weight_decay=0.05, grad_clip=1.0`
  - 最终验证损失（5000 步）：≈1.734（日志见 `artifacts/run_train_20260201-200613/exp_log.jsonl`）
  - 曲线要点：`val_loss` 在 2400→4800 步区间持续下降，最低点 ≈1.725（4800 步），5000 步略回升到 ≈1.734（总体稳定）
  - 复现命令：`MAX_STEPS=5000 WARMUP_ITERS=2000 COSINE_CYCLE_ITERS=5000 MAX_LR=6e-4 MIN_LR=3e-4 WEIGHT_DECAY=0.05 VOCAB_SIZE=10000 bash run_training.sh train artifacts/tinystories_tokens/train.npy artifacts/tinystories_tokens/valid.npy mps`
- 运行 B（基线）：`max_lr=1e-4, min_lr=3e-5, warmup=500, cosine_cycle_iters=5000, weight_decay=0.01`
  - 在同等设定下，5000 步附近 `val_loss` 明显高于运行 A（此前记录≈2.15）
  - 复现命令：`MAX_STEPS=5000 WARMUP_ITERS=500 COSINE_CYCLE_ITERS=5000 MAX_LR=1e-4 MIN_LR=3e-5 WEIGHT_DECAY=0.01 VOCAB_SIZE=10000 bash run_training.sh train artifacts/tinystories_tokens/train.npy artifacts/tinystories_tokens/valid.npy mps`
- 运行 C（更大步长试跑）：`max_lr=2e-3, min_lr=1.2e-3`（短试跑 800 步）
  - 观察到训练初期 `train_loss` 快速变化、对学习率敏感，需要更长 warmup 与更强正则以维持稳定（后续补充完整曲线）
  - 复现命令：`MAX_STEPS=800 WARMUP_ITERS=100 COSINE_CYCLE_ITERS=5000 MAX_LR=2e-3 MIN_LR=1.2e-3 VOCAB_SIZE=10000 bash run_training.sh train artifacts/tinystories_tokens/train.npy artifacts/tinystories_tokens/valid.npy mps`

=== 最佳学习率与目标达成
- 在本机 MPS 低资源目标下，运行 A 的设定在 5000 步达成 `val_loss≈1.734`，满足“低资源将目标提高到 ≤2.00”的要求；相较基线 `1e-4`，更接近“稳定边缘”，收敛更快、曲线更低。
- 综合结论：在该模型与数据规模下，靠近稳定边缘的较大 `max_lr`（并配合更长 warmup 与适度 `weight_decay`）能显著提升 5000 步内的验证表现；但需要监控尾段回升并适配正则。

=== “稳定边缘”分析（b）
- 随着学习率增大，早期下降速度变快，但曲线更易出现波动甚至发散；A 设定在 4800 步前后出现轻微回升，提示已逼近稳定边缘；再上探到 `2e-3` 时对 warmup/正则的依赖显著增强，若不调参易不稳。
 - 本机低资源最佳设定（实验 A）：`batch_size=32`，`max_lr=6e-4`，`min_lr=3e-4`，`warmup=2000（≈总步数40%）`，`cosine_cycle_iters=5000`，`weight_decay=0.05`，`beta2=0.95`，`grad_clip=1.0`；在 5000 步 `val_loss≈1.734`。在此基础上可做小幅微调；更大的 batch 建议按线性规则上调 `max_lr` 并以短试跑验证稳定性。

=== 低资源提示与实现细节
- 我们严格遵循 CS336 低资源建议：总 tokens ≈40,960,000、MPS 不启用 TF32，必要时可在 sweep 中用 `torch.compile(backend="aot_eager")` 优化后向。
- 实验日志统一写入 `artifacts/run_train_<timestamp>/exp_log.jsonl`，可直接用于绘制多条学习曲线；本节以运行 A 的完整曲线为主，基线与高学习率短跑用于对比说明稳定边缘。
=== 学习率曲线（实验 A）
#image("artifacts/run_train_20260201-200613/learning_curves.svg", width: 80%)
=== Batch size 对比（A 配置，固定 LR，顺序执行）
#image("artifacts/run_train_20260201-224606/learning_curves.svg", width: 80%)
#image("artifacts/run_train_20260201-235644/learning_curves.svg", width: 80%)
==== 结果与分析
- bs=32（A）：`val_loss≈1.7336@5000`
- bs=64（A）：`val_loss≈1.6575@5000`
- bs=128（A）：`val_loss≈1.6083@5000`
- 固定较高学习率（A 配置）下，批次增大使曲线更平滑且最终 `val_loss` 更低，但需要更长 warmup 和适度正则保持稳定；在本机低资源下，`bs=128` 表现最佳。
==== 叠加对比图（val_loss）
#image("artifacts/batch_size_overlay.svg", width: 80%)

== Problem (generate)：Generate text（1 分）
=== 设置与脚本
- 使用已训练检查点：`artifacts/run_train_20260201-235644/lm.ckpt`（A 配置，bs=128）
- 解码参数：`temperature=0.9`，`top_p=0.95`，`max_new_tokens=256`，遇到 `<|endoftext|>` 立刻停止
- 生成脚本：`scripts/generate_text_from_ckpt.py`（调用 `cs336_basics/decoding.py:23-51`）

=== Prompt
- `Once upon a time, in a small village, a kind robot named Lumo`

=== 输出（直出文本）
Once upon a time, in a small village, a kind robot named Lumo was always alert and ready to help. One day, she saw a big pot in the kitchen. She wanted to fill the pot with something yummy.
Resodded, she found a lot of delicious food. She ate and ate until she was full. Then she started to stir something in the pot. It was a hot soup! The pot of soup smelled so good!
The tiny rat came out of the pot. It was very hungry. She went back to the village to eat the soup. The people in the village saw the soup and were very happy. The cauliflower said, "Thank you, Crida! I love my delicious soup!" They all sat together and ate the delicious soup.
<|endoftext|>

=== 简短评论
- 流畅度：整体连贯，句式与词汇简单，符合 TinyStories 风格。个别用词（如 “Resodded”）存在低频拼写/词义噪声。
- 影响因素（至少两点）：
  - 采样参数：`temperature/top_p` 直接影响多样性与连贯性；较高 `temperature`/`top_p` 增加新颖性，但更易出现不常见词或逻辑跳跃；降低可使文本更稳但可能重复。
  - 训练配置与步数：学习率调度对齐步数（warmup+余弦）与总训练步（5000）影响验证损失与生成质量；本机 A 配置下 bs=128 的验证更低，生成文本更稳。
  - 上下文长度与提示设计：`context_length=256` 限制模型可见的历史；更具体的 prompt（角色、目标、场景）通常提升连贯性与细节。

== Problem (layer_norm_ablation)：Remove RMSNorm 并训练（1 分）
=== 设置
- 架构保持一致，仅移除所有 RMSNorm（块内两处与顶层一处）；训练数据与上下文长度保持不变
- 旧最优学习率（A 配置）：`max_lr=6e-4, min_lr=3e-4, warmup=2000, cosine_cycle_iters=5000, weight_decay=0.05, bs=128`
- 降低学习率试跑（2000 步）：`max_lr=2e-4, min_lr=6e-5, warmup=200, bs=128`

=== 学习曲线
#image("artifacts/run_train_20260208-120254/learning_curves.svg", width: 80%)

=== 简短评论
- 在旧最优学习率下，移除 RMSNorm 的训练并未发散，曲线整体平滑但最终 `val_loss≈1.689@5000` 明显高于有 RMSNorm 的结果（`≈1.608@5000`），表现更差
- 降低学习率后训练同样稳定，但收敛速度更慢，短试跑显示下降趋势但预计跑满后仍难达到含 RMSNorm 的最佳验证损失
- RMSNorm 通过通道尺度归一化提升数值稳定性与收敛效率；移除后“稳定边缘”更窄、最佳学习率区间更保守，性能与收敛速度均受影响

== Problem (pre_norm_ablation)：Post-norm 与训练（1 分）
=== 设置
- 脚本：`2.post_norm.sh:3`（`NORM_STYLE=post`，其它超参与 TinyStories A 配置一致，bs=128）
- 运行结果目录：`artifacts/run_train_20260208-170842`
- 修改：将块内归一化从 pre-norm 改为 post-norm，次序为 `y1 = LN(x + Attn(x))`，`y = LN(y1 + FFN(y1))`

=== 学习曲线与对比
#image("artifacts/run_train_20260208-170842/learning_curves.svg", width: 80%)
#image("artifacts/pre_vs_post_norm_170842_overlay.svg", width: 80%)

=== 简短评论
- Post-norm 在当前较高 LR（A 配置）下可正常训练，但相较 pre-norm 对 warmup/正则更敏感，稳定边缘更窄
- 与 pre-norm 的对比显示，Post-norm 的验证曲线下降更依赖稳健的调度，适度降低 LR 能改善稳定性；综合而言，pre-norm 更适合深层网络与更快稳态收敛

== Problem (no_pos_emb)：RoPE vs NoPE（1 分）
=== 设置
- 脚本：`3.nope.sh:2`（`NO_POS_EMB=1` 关闭 RoPE），其它超参与 TinyStories A 配置一致（bs=128）
- 运行结果目录：`artifacts/run_train_20260208-173722`
- 实现：`TransformerLM(use_rope=False)` → `MultiHeadSelfAttention` 中不应用 RoPE（仍保留因果掩码）

=== 学习曲线与对比
#image("artifacts/run_train_20260208-173722/learning_curves.svg", width: 80%)
#image("artifacts/rope_vs_nope_173722_overlay.svg", width: 80%)

=== 简短评论
- NoPE 在该小模型与统一语域（TinyStories）下，前期斜率与 RoPE 接近，验证曲线差距不大；因果掩码与内容相似度即可学习到短程统计关系
- RoPE 在长上下文与跨句关联上更占优：当上下文加长或语料更复杂（如 OWT），NoPE 的平台更高、生成更易重复或断连；相对位置编码提供的距离刻度能显著改善泛化与顺序敏感度

== Problem (swiglu_ablation)：SwiGLU vs. SiLU（1 分）
=== 设置
- SiLU 结果目录：`artifacts/run_train_20260208-175021`；SwiGLU 对照：`artifacts/run_train_20260201-235644`
- 参数量匹配：SiLU 的隐藏维设为 `≈1.5×d_ff`，令两者参数量接近（`≈3·d_model·d_ff`）
- 其它架构与超参与 A 配置一致（bs=128，`max_lr=6e-4, min_lr=3e-4, warmup=2000, cosine=5000`）

=== 学习曲线与对比
#image("artifacts/run_train_20260208-175021/learning_curves.svg", width: 80%)
#image("artifacts/swiglu_vs_silu_175021_overlay.svg", width: 80%)

=== 简短评论
- 在匹配参数量条件下，SwiGLU 的门控乘法带来更强的非线性表达与更低的验证损失，尤其在较高学习率下更稳
- SiLU 需要更保守的学习率与较长 warmup 才能接近最优；早期趋势接近，但最终 `val_loss` 与稳定性通常略逊于 SwiGLU
