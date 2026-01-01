#set page(numbering: "1")
#set heading(numbering: "1.")
#set text(font: "Noto Serif CJK SC")
#show heading: set text(font: "Noto Serif CJK SC")

= CS336 Assignment 1 Writeup
作者：蔡志威
日期：2026-01-01

#outline()

== Problem 1 概述
在本题中，请清晰给出问题背景与目标，并用简短段落说明你的思路与结论。

=== 解答与推导
给出主要结论与关键推导步骤。
$
  alpha_t = alpha_min + 0.5 (alpha_max - alpha_min)
  (1 + cos(pi * (t - T_w) / T_c)),\ t >= T_w
$

=== 图与结果（可选）
如需展示曲线或示意图：
#figure(
  [占位图：请将图片路径替换为 image("your-image.png") 或删除此段落],
  caption: [Learning rate curve],
) <fig:curve>
如图 @fig:curve 所示。

=== 表格（可选）
#table(
  columns: 3,
  [Metric], [Value], [Note],
  [LR], [0.003], [Max],
  [Warmup], [1000], [Iters],
)

== Problem 2 概述
简述题目与方法。根据需要添加若干小节，如“模型”、“算法”、“复杂度分析”等。

=== 解答
在此填写详细答案与必要的推导、示例或反例。

== Problem 3（如有）
同上结构组织内容。

== 讨论与局限（可选）
说明结果的意义、适用范围、潜在局限与改进方向。

== References（如有）
- 参考文献示例：作者, 标题, 期刊/会议, 年份。

== 附录（可选）
放置冗长的推导、补充图表或附加说明。
