# -*- coding: utf-8 -*-
"""论文结构机器核验（PaperSpine 六维评估的取证脚本）。
只检查结构一致性，不修改 tex。
"""
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

import glob
import os

# 目标文件：优先命令行参数；否则自动取 post/ 下版本号最大的
# DvM_Paper_Anonymous_v*.tex（避免像 v33 那样改了文件却仍在核验旧版本）
if len(sys.argv) > 1:
    TEX = sys.argv[1]
else:
    cands = glob.glob(r"D:/spikergnn/DvM/post/DvM_Paper_Anonymous_v*.tex")
    TEX = max(cands, key=lambda p: (int(re.search(r"_v(\d+)\.tex$", p).group(1)), os.path.getmtime(p))) if cands \
        else r"D:/spikergnn/DvM/post/DvM_Paper_Anonymous.tex"

t = open(TEX, encoding="utf-8").read()
print(f"核验目标: {TEX}")

print("=" * 74)
print("A. 图表 / 定理 编号与引用一致性")
print("=" * 74)

# --- table / figure 定义 ---
tab_defs = [(m.group(1), m.start()) for m in re.finditer(r"\\label\{(tab:[^}]+)\}", t)]
fig_defs = [(m.group(1), m.start()) for m in re.finditer(r"\\label\{(fig:[^}]+)\}", t)]
# --- 定理环境 ---
thm_envs = re.findall(r"\\begin\{(theorem|theorem\*|corollary|lemma| proposition)\}", t)

# --- 顺序编号（tab:xxx / fig:xxx 在源文件中出现顺序 = 实际编号） ---
seq_tab, seq_fig = [], []
for name, pos in sorted(tab_defs + fig_defs, key=lambda x: x[1]):
    pass
order = sorted([(p, n) for n, p in tab_defs] + [(p, n) for n, p in fig_defs])
for i, (p, n) in enumerate(order, 1):
    (seq_tab if n.startswith("tab:") else seq_fig).append((i, n))

print(f"\n表定义 {len(tab_defs)} 个：")
for i, n in seq_tab:
    print(f"  Table {i:2d}  {n}")
print(f"\n图定义 {len(fig_defs)} 个：")
for i, n in seq_fig:
    print(f"  Figure {i:2d}  {n}")

# --- 引用检查：\ref{} 与 \label{} ---
refs = set(re.findall(r"\\ref\{([^}]+)\}", t))
labels = set(re.findall(r"\\label\{([^}]+)\}", t))
bad_ref = sorted(refs - labels)
print(f"\n\\ref 指向不存在的 \\label： {bad_ref if bad_ref else '无'}")

# --- 表格/图是否都被正文引用 ---
unref_tab = [n for _, n in seq_tab if n not in refs]
unref_fig = [n for _, n in seq_fig if n not in refs]
print(f"未被 \\ref 引用的表： {unref_tab if unref_tab else '无'}")
print(f"未被 \\ref 引用的图： {unref_fig if unref_fig else '无'}")

# --- 硬编码的 "Table N" 文字（应与自动编号一致） ---
hard_tab = re.findall(r"Tables?[~\s]*(\d+)", t)
hard_ref = re.findall(r"\\eqref\{[^{}]+\}", t)
print(f"\n正文硬编码 'Table N' 次数： {len(hard_tab)}  -> {sorted(set(hard_tab, key=int)) if False else sorted(set(int(x) for x in hard_tab))}")

print("\n" + "=" * 74)
print("B. 定理 / 引理 /  corollary 声明")
print("=" * 74)
for m in re.finditer(r"\\(?:begin\{(theorem\*?|corollary\*?|lemma\*?)\}|textbf\{(Theorem|Corollary|Lemma)\s*\d+[^}]*\})", t):
    s = m.group(0)
    ln = t[:m.start()].count("\n") + 1
    print(f"  line {ln:4d}  {s[:70]}")
# 环境版
env_theorem = [m.start() for m in re.finditer(r"\\begin\{theorem\*?\}", t)]
env_cor = [m.start() for m in re.finditer(r"\\begin\{corollary\*?\}", t)]
print(f"\n  theorem 环境数：{len(env_theorem)}   corollary 环境数：{len(env_cor)}")
thm_head = [m.start() for m in re.finditer(r"\\textbf\{Theorem\s*1", t)]
print(f"  正文 'Theorem 1' 出现次数：{len(thm_head)}  -> 是否为不同实体（需检查）")

print("\n" + "=" * 74)
print("C. 章节标题与是否有正文内容")
print("=" * 74)
secs = [(m.group(1), m.start()) for m in re.finditer(r"^\\section\{([^}]*)\}", t, re.M)]
subs = [(m.group(1), m.start()) for m in re.finditer(r"^\\subsection\{([^}]*)\}", t, re.M)]
for i, (name, pos) in enumerate(secs, 1):
    end = secs[i][1] if i < len(secs) else len(t)
    body = len([l for l in t[pos:end].split("\n") if l.strip() and not l.startswith("\\")])
    print(f"  {i}. {name[:60]}")
# subsection 内容量
positions = sorted([p for _, p in secs] + [p for _, p in subs])
for i, (name, pos) in enumerate(subs, 1):
    nxt = subs[i][1] if i < len(subs) else len(t)
    body = len([l for l in t[pos:nxt].split("\n") if l.strip() and not l.startswith("\\")])
    flag = "   <== 无正文" if body == 0 else ""
    print(f"     - {name[:56]:58s} 正文行={body}{flag}")

print("\n" + "=" * 74)
print("D. 引用完整性")
print("=" * 74)
cites = set(re.findall(r"\\cite[tp]?\{([^}]+)\}", t))
cites_flat = set()
for c in cites:
    for k in c.split(","):
        cites_flat.add(k.strip())
bibs = set(re.findall(r"\\bibitem\[[^\]]*\]\{([^}]+)\}", t))
print(f"  \\cite 唯一 key 数：{len(cites_flat)}")
print(f"  \\bibitem 数：{len(bibs)}")
print(f"  被引用但未定义 bibitem： {sorted(cites_flat - bibs) if (cites_flat - bibs) else '无'}")
print(f"  定义了但从未被引用的 bibitem： {sorted(bibs - cites_flat) if (bibs - cites_flat) else '无'}")

print("\n" + "=" * 74)
print("E. 缩写首次出现检查")
print("=" * 74)
abbrs = ["SLTT", "DvM", "BPTT", "SNN", "MSPE", "STMP", "RSGD", "LIF", "GIN", "GCN"]
for a in abbrs:
    # 找第一次出现的上下文（去掉定义行本身）
    hits = [(m.start(), t[:m.start()].count("\n") + 1) for m in re.finditer(r"\b" + a + r"\b", t)]
    if hits:
        ln = hits[0][1]
        print(f"  {a:6s} 首次 line {ln}")

print("\n" + "=" * 74)
print("F. AI 痕迹 / 占位符扫描")
print("=" * 74)
for kw in ["WorkBuddy", "ChatGPT", "Claude", "LLM", "大模型", "人工智能助手", "ChatGPT-"]:
    n = t.count(kw)
    if n:
        print(f"  [!] 命中 AI 痕迹词 '{kw}' x{n}")
print("  AI 痕迹扫描完成")

print("=" * 74)
