#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
引用内容审计脚本 —— PDF 原文基准版
================================================================
与"形式核验"(key 闭环 / DOI 可解析 / LaTeX 可编译) 不同，本脚本做**内容核验**。

幻觉引用的特征是「格式全对、内容全假」：
  key 闭环 ✓  格式规范 ✓  能编译 ✓   但 标题/作者/年份/arXiv编号 是编造的 ✗

本脚本以 DvM/paper/ 目录下**真实 PDF 的原文文本**为唯一基准，
对 thebibliography 中每条 \\bibitem 逐字段核对：
  1) arXiv 编号：bibitem 声明的编号  vs  PDF 封面扫描到的 arXiv ID
  2) 作者：bibitem 列出的每个作者姓氏，是否真出现在 PDF 原文中（找不到的=疑似编造）
  3) 标题：真实标题的实词 token 在 PDF 原文中的覆盖率
  4) 文件名一致性：paper/ 里文件名声称的文献，与 PDF 实际内容是否对得上
     （典型案例：Gu2023_SLTT.pdf 实际是一篇天文学论文）

完全离线，不依赖任何 API。

用法:
    python audit_pdf.py [tex路径] [--paper-dir D:/spikergnn/DvM/paper] [--out 报告.md]
默认 tex: post/DvM_Paper_Anonymous.tex
"""
import re
import os
import unicodedata
import sys
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor

PDFTOTEXT_CANDIDATES = [
    r"D:/texlive/2026/bin/windows/pdftotext.exe",
    r"D:/texlive/2026/bin/windows/pdftotext",
    "pdftotext",
    "pdftotext.exe",
]


def find_pdftotext():
    """定位 pdftotext。注意：Git Bash 的 /d/... 路径 Python 不认，必须转 Windows 盘符。"""
    import glob
    for p in PDFTOTEXT_CANDIDATES:
        if os.path.exists(p):
            return p
    hits = glob.glob(r"D:/texlive/*/bin/windows/pdftotext.exe")
    if hits:
        return hits[0]
    for p in os.environ.get("PATH", "").split(os.pathsep):
        for exe in (os.path.join(p, "pdftotext.exe"), os.path.join(p, "pdftotext")):
            if os.path.exists(exe):
                return exe
    return None


PDFTOTEXT = find_pdftotext()


def extract_text(pdf_path, first_page_only=False):
    """用 pdftotext 抽取 PDF 文本。first_page_only=True 时只抽第 1 页（用于提标题）。"""
    if not PDFTOTEXT:
        raise RuntimeError("未找到 pdftotext 可执行程序")
    cmd = [PDFTOTEXT, "-enc", "UTF-8"]
    if first_page_only:
        cmd += ["-f", "1", "-l", "1"]
    cmd += [pdf_path, "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120)
        return out.stdout.decode("utf-8", "ignore")
    except Exception:
        return ""


# ---------------------------------------------------------------- 清洗 / 归一化

def clean_latex(s):
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    # 通用重音字符： \'e  \`o  \"o  \v{c}  \c{c}  \u{a}  ...
    s = re.sub(r"\\['`^~\"](?:\{)?([a-zA-Z])(?:\})?", r"\1", s)
    s = re.sub(r"\\(?:v|c|u|H|t|d|b|r)(?:\{)?([a-zA-Z])(?:\})?", r"\1", s)
    s = re.sub(r"\\&", "&", s)      # 注意：正则里 \& 是"字面 &"，必须写 \\\& 才匹配反斜杠
    s = re.sub(r"\\_", "_", s)
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def fold(s):
    """Unicode 折叠：先 NFKD 展开，去掉组合记号，再只留 ASCII 字母数字。

    关键：pdftotext 抽出的文本常含**组合型重音**（NFD），如 "Veličković" 实际是
    "Velic◌̌kovic◌́"。若直接用 [^a-z] 剥离，基字母 'c' 会连着重音一起被吞掉，
    变成 "velikovi"，导致作者"查不到"的假阳性。
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.encode("ascii", "ignore").decode("ascii")


def norm_name(s):
    """姓名归一化：去大小写、变音符号、空格连字符（返回纯字母串）"""
    s = fold(s).lower()
    s = s.replace(" ", "")          # "Lopez de Compadre" → "lopezdecompadre"
    s = s.replace("-", "")
    for a, b in [("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"),
                 ("ü", "u"), ("ä", "a"), ("ö", "o"), ("ñ", "n"), ("ç", "c"),
                 ("š", "s"), ("ž", "z"), ("č", "c"), ("ć", "c"), ("ĉ", "c"),
                 ("ł", "l"), ("ø", "o"), ("å", "a"), ("ņ", "n"), ("ś", "s"),
                 ("ğ", "g"), ("ı", "i"), ("đ", "d"), ("æ", "a"), ("œ", "o")]:
        s = s.replace(a, b)
    return re.sub(r"[^a-z]+", "", s)


def tokens(s):
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return [x for x in s.lower().split() if len(x) > 2]


def compact(s):
    """紧凑归一化：去所有非字母数字，解决 arXiv 模板"逐字母加空格"的大写标题
    （如 "T RAINING S PIKING N EURAL N ETWORKS" → "trainingspikingneuralnetworks"）"""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def kw_in(kw, text):
    c = compact(kw)
    return len(c) >= 8 and c in compact(text)


_WORD_RE = {}


def has_word(norm_text, name):
    """姓氏/词是否在原文中出现 —— **词边界**匹配。

    为什么不能用子串：norm 文本里 "through" 包含 "lou"、"click" 包含 "xu"、
    "human" 包含 "hu"，短姓氏会大面积假阳性，导致 bibitem 根本通不过姓氏门槛。
    为什么不用普通 \\b 加 $：PDF 里作者常带编号/星号，如 "Weihua Hu1," 、
    "Keyulu Xu *"，补一个 "(?![a-z])" 后视字符即可同时满足二者。
    norm 基准文本必须保留空格（姓名内部空格已被 norm_name 吃掉，不会粘连）。
    """
    n = norm_name(name)
    if len(n) < 2:
        return False
    rx = _WORD_RE.get(n)
    if rx is None:
        rx = _WORD_RE[n] = re.compile(r"\b" + re.escape(n) + r"(?![a-z])")
    return bool(rx.search(norm_text))


def fname_words(stem):
    """从文件名里取出候选姓氏。

    paper/ 的文件名把姓氏连写成一个串："BecigneulGanea2019_..." 直接
    norm_name 会粘成 "becigneulganea"，自然在原文里查不到 —— 这是假阳性。
    这里在"小写→大写"边界切分，逐个候选去原文里找。
    """
    base = re.split(r"[_\-]", os.path.splitext(stem)[0])[0]
    base = re.sub(r"\d+$", "", base)
    parts = re.findall(r"[A-Z][a-z]+|[A-Z]+(?![a-z])|[a-z]+", base)
    words = [p for p in parts if len(norm_name(p)) >= 3]
    return words or [base]


AUTHOR_ZONE = 6000


def authored_by(rec, name, zone_chars=AUTHOR_ZONE):
    """姓氏是否在**署名区**（正文前 zone_chars 字符）里作为作者出现。

    为什么不能查全文：PDF 自带参考文献会把别文的作者带进来。实测
    Deng2021 正文写着 "(Roy et al., 2019)"、"Han & Roy (2020)"，
    若按全文匹配，bibitem Roy19(Roy/Jaiswal/Panda) 会被"匹配"到 Deng2021
    上，结论全错。故限定署名区，并显式排除引用语境。
    """
    zone = rec["norm"][:zone_chars]
    zonep = rec["normp"][:zone_chars]
    # 复姓/连名（"De Sa"、"Lopez de Compadre"）：整体拼写法在 PDF 里未必连着写，
    # 必须按空格拆开逐个试，否则 "Desa"、"Lopezdecompadre" 永远查不到 → 假阳性。
    cands = []
    if len(norm_name(name)) >= 3:
        cands.append(name)                       # 整体（"Lopez de Compana"）
    cands += fname_words(name)                   # 连写姓氏 "BecigneulGanea"
    cands += [t for t in name.split() if len(norm_name(t)) >= 2]
    for c in list(dict.fromkeys(cands)):
        n = norm_name(c)
        if len(n) < 2:
            continue
        for m in re.finditer(r"\b" + re.escape(n) + r"(?![a-z])", zone):
            # 上下文窗口要够长：normp 把字母换成了空格，短窗口（如 8 字符）
            # 在 "roy et al., 2006" 处整段都是空格，会漏掉 "et al" 排除规则。
            c = rec["ctx"]
            pre = c[max(0, m.start() - 12):m.start()]
            post = c[m.end():m.end() + 30]
            if pre.rstrip().endswith("&"):        # "Han & Roy (2020)"
                continue
            if pre.rstrip().endswith("(") or pre.rstrip().endswith("["):
                continue                           # "(Roy et al., 2019)"
            # 只排除 "et al."；若连 "," ")" 一起排除，署名后的单位编号
            # "Meng1,2" 也会误伤，真作者会被当成引用剔掉。
            if re.match(r"\s*et\.?\s*al", post):
                continue                           # "Roy et al."
            return True
    return False


def fam_ok(rec, name, zone_chars=AUTHOR_ZONE):
    """候选姓氏任一命中即可。"""
    return authored_by(rec, name, zone_chars=zone_chars)


def tok_hit(rec, t):
    """标题 token 命中判定。
    arXiv 模板会把标题逐字母加空格（"T RAINING S PIKING..."），
    normc 是无空格版，长 token 用它做紧凑匹配即可穿透。"""
    if len(t) >= 4 and compact(t) in rec["normc"]:
        return True
    return has_word(rec["norm"], t)


# ---------------------------------------------------------------- tex 解析

def parse_bibitems(tex):
    m = re.search(r"\\begin\{thebibliography\}.*?(?=\\section\{Reproducibility\}|\Z)",
                  tex, re.S)
    if not m:
        return []
    block = m.group(0)
    items = []
    for bm in re.finditer(r"\\bibitem\[([^\]]*)\]\{([^}]*)\}(.*?)(?=\\bibitem\[|\Z)",
                          block, re.S):
        label, key, raw = bm.groups()
        body = clean_latex(raw)
        authors, title, year = split_authors_year(body)
        items.append(dict(key=key, label=label, body=body, raw=raw,
                          authors=authors, title=title, year=year))
    return items


VENUE_RE = re.compile(
    # ① 显式出处标记（正文里的 "In " / "Paper presented at" / "Journal of" ...）
    #    注意必须 IGNORECASE：bibitem 里写的是 "Frontiers in Neuroscience"，
    #    首字母不大写，原大小写敏感版匹配不到，会把期刊名一起算进标题，
    #    稀释标题覆盖率 → Wu18STBP 之类的条目标题覆盖率跌破阈值 → 误报"无匹配"。
    r"\s(?:In\s|Paper presented at\s|Published as\s|https?://|doi\.org|"
    r"Proceedings of|arXiv preprint|Journal of|Frontiers in|Advances in|pp\.\s)"
    # ② LaTeX 出处：journal 名一律用 \textit{...} 包着
    r"|\s\$\textit|\s\\textit\{|\s\\text\{"
    # ③ 句首大写开头的刊物名，如 "compounds. Journal of Medicinal Chemistry"
    r"|\.\s+(?:Journal|Frontiers|Proceedings|Advances|Nature|IEEE|ACM)\b",
    re.I)


def split_authors_year(body):
    """从 bibitem 正文中切出 作者 / 标题 / 年份。

    注意：不能简单取年份后到 "." 为止 —— 会议/期刊出处 ("In Advances in ...")
    也会被算进标题，导致标题 token 与一堆无关 PDF 产生误匹配。
    """
    ym = re.search(r"\s\((\d{4})", body)
    if ym:
        author_part, year = body[:ym.start()], ym.group(1)
        rest = body[ym.end():].lstrip(" .)")     # 年份括号后的 ")" 也要去掉
    else:
        author_part, year, rest = body, "?", body

    # 标题 = 年之后的文本，止于第一个"出处"标记或首个 ". "
    m = VENUE_RE.search(rest)
    if m:
        title_part = rest[:m.start()]
    else:
        sm = re.search(r"\.\s+[A-Z]", rest)
        title_part = rest[:sm.start()] if sm else rest
    title_part = title_part.strip(" .")

    authors = re.split(r",\s*|\s+&\s+|\s+and\s+", author_part.strip())
    authors = [a.strip(" .&,") for a in authors if a.strip(" .&,")]

    # 省略号表示 et al.，其后的作者无法核验，截断
    out = []
    for a in authors:
        if a.startswith("..."):
            break
        out.append(a)
    return out, title_part, year


# ---------------------------------------------------------------- PDF 索引

def pdf_arxiv_id(full_text):
    m = re.search(r"arXiv[:\s]*(\d{4}\.\d{4,5})v?\d?", full_text, re.I)
    return m.group(1) if m else ""


def pdf_title(first_page):
    """从首页文本粗略提取标题行。
    过滤掉带编号(作者 affiliations)、邮箱、页眉页脚的行，取最长的说明性行。
    """
    lines = [l.strip() for l in first_page.splitlines()[:40]]
    skip = re.compile(
        r"^(arxiv|preprint|submitted|accepted|published|©|all rights|"
        r"received|abstract|keywords| forthcoming|under review|"
        r"contents|index|1\s*$|monte carlo)", re.I)
    cands = []
    for l in lines:
        if not l or len(l) < 8 or len(l) > 200:
            continue
        if skip.match(l):
            continue
        if "@" in l:                       # 邮箱行
            continue
        if re.search(r"\d", l):            # 带编号的头像 Affiliation 行
            continue
        if l.endswith(".") and len(l.split()) < 6:
            continue
        cands.append(l)
    if not cands:
        return ""
    return max(cands, key=lambda x: (len(x.split()), len(x)))


def build_paper_index(paper_dir, workers=8):
    pdfs = sorted(f for f in os.listdir(paper_dir) if f.lower().endswith(".pdf"))
    idx = {}

    def work(fname):
        path = os.path.join(paper_dir, fname)
        try:
            full = extract_text(path, first_page_only=False)
            p1 = extract_text(path, first_page_only=True)
        except Exception:
            full, p1 = "", ""
        low = full.lower()
        # norm / normp 必须**逐字符对齐**：前者只把字母数字换成空格，
        # 后者保留标点。若用 re.sub 折叠连续空格，norm 会与原始串错开，
        # 引用语境窗口读到错位文本，排除规则反而在真作者身上误伤。
        _f = fold(low)
        _norm = "".join(ch if ch.isalnum() else " " for ch in _f)
        _normp = "".join(" " if ch.isalnum() else ch for ch in _f)
        # ctx 与原串逐字符对齐、且**字母和标点都保留**，供引用语境判定用。
        # 不能用 normp：它把字母换成了空格，"et al" 这类上下文就消失了。
        _ctx = _f
        return fname, dict(
            file=fname,
            arxiv_id=pdf_arxiv_id(full),
            title=pdf_title(p1) or pdf_title(full),
            text_head=full[:6000],
            lower=low,
            # 词边界匹配基准：**必须保留空格**，否则姓名粘连成
            # "michaelmbronstein"，\b 词边界全部失效
            norm=_norm,
            normp=_normp,
            ctx=_ctx,
            normc=re.sub(r"[^a-z0-9]+", "", _f),   # 无空格版，用于标题紧凑匹配
            nchars=len(full),
        )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fname, rec in ex.map(work, pdfs):
            idx[fname] = rec

    # 文件名 vs 内容 一致性检查
    # 这类错误最隐蔽：文件叫 "Eshraghian2023_XXX.pdf"，内容却完全另一篇。
    # 判据：文件名里的首作者姓氏是否真的出现在 PDF 原文中。
    for fname, rec in idx.items():
        stem = os.path.splitext(fname)[0]
        m = re.match(r"^([A-Za-z'\-]+?)(\d{2,4})_", stem)
        if not m:
            rec["name_issue"] = ""
            continue
        fam = m.group(1)
        in_text = fam_ok(rec, fam)
        rec["name_fam"] = m.group(1)
        rec["name_issue"] = "" if in_text else (
            f"文件名声称作者 {m.group(1)}，但 PDF 原文中查不到该姓氏"
            f"（PDF 首页实际内容: {rec['title'][:60]}）")
    return idx


# ---------------------------------------------------------------- 审计

def vocab(tex):
    """所有 bibitem 标题的 token 集合，作为稀有度统计的候选词表。"""
    vs = set()
    for it in parse_bibitems(tex):
        vs.update(t for t in tokens(it["title"]) if len(t) >= 2)
    return vs


def hit_tokens(rec, vocab):
    """该 PDF 命中的 token 集合（长词走紧凑匹配，短词走词边界匹配）。"""
    return {t for t in vocab if tok_hit(rec, t)}


def build_df(index, v):
    """统计 token 在 **PDF 语料**（而非 bibitem 标题）中的文档频次。

    早期版本按 bibitem 标题统计，结果 "hyperbolic" 被当成 df=4 的罕见词，
    但它在 paper/ 的十来篇都出现 → 加权几乎无效，Lin23 误配到 SLTT。
    稀有度必须以实际比对的那份语料为准。
    """
    df = {}
    for fname, rec in index.items():
        for t in hit_tokens(rec, v):
            df[t] = df.get(t, 0) + 1
    return df


def match_pdf(item, index, ctx=None):
    """把 bibitem 匹配到某个 PDF。

    匹配必须是「双因子」的，只靠标题 token 会把一堆文献都误配到同一篇
    （因为 "Advances in Neural Information Processing Systems" 之类
     出处词到处都是）。这里要求：
        arXiv 编号完全一致  OR   (首作者姓氏命中 AND 标题覆盖率 >= 0.6)
    """
    claimed_arxiv = ""
    m = re.search(r"arXiv[: ]*([0-9]{4}\.[0-9]{4,5})", item["body"], re.I)
    if m:
        claimed_arxiv = m.group(1)

    # 1) arXiv 编号精确匹配（最可靠）
    if claimed_arxiv:
        for fname, rec in index.items():
            if rec["arxiv_id"] == claimed_arxiv:
                return fname, f"arXiv编号={claimed_arxiv}"

    # 2) 双因子匹配
    # 关键：覆盖率必须按**文档频次加权**。
    # 若用原始覆盖率，只看 len>2 的 token，标题 "Deep residual learning for
    # spiking neural networks" 会被压缩成 [deep, residual, learning]，而这三个
    # 词在 paper/ 里每篇 SNN 论文都有 → 所有 PDF 覆盖率都是 1.00 → 排序失效，
    # Fang21 会随机落到 Bronstein 上。改为 1/(1+df) 加权后，
    # "residual" 这类罕见词权重高、"spiking/neural/network" 这类泛词近乎归零，
    # 只有真正同一篇才能拿到高加权覆盖率。
    # 纯数字 token 要滤掉：页码 "490--501" 会被当成标题词，
    # 这些词在任何 PDF 里都查不到 → 稀有词约束永远不过 → 误报"无匹配"。
    tk = [t for t in tokens(item["title"])
          if len(t) >= 2 and not t.isdigit()]
    # 姓氏 = 作者串中逗号前的第一段（APA 格式 "Eshraghian, J. K."）
    first_fam = ""
    for a in item["authors"]:
        if a.split(","):
            first_fam = a.split(",")[0].strip()
            break
    fam = first_fam
    dfw = ctx if ctx is not None else {}
    best, best_score, best_cov = None, 0.0, 0.0
    if tk:
        for fname, rec in index.items():
            w_tot = w_hit = 0.0
            for t in tk:
                w = 1.0 / (1.0 + dfw.get(t, 0))
                w_tot += w
                if tok_hit(rec, t):
                    w_hit += w
            cov = w_hit / w_tot if w_tot else 0.0
            if cov < 0.45:
                continue
            # 首作者姓氏校验：对不上则绝不可能是这篇（对短姓氏也强制校验）
            # 姓氏门槛仍用 6000 字，并靠引用语境排除（见 authored_by 注释）。
            # 早期试过收紧到 1500 字，反而把 Kag19、MUTAG 这类署名排在
            # 页面靠后的老论文 PDF 误判成"无匹配"。
            if not fam or not fam_ok(rec, fam):
                continue
            # 稀有词（df<=3）必须命中：标题里 "SLTT" 这种专名一旦落空，
            # 说明根本不是同一篇，不能被 "spiking/neural/network" 之类
            # 泛词的加权覆盖率蒙混过关（Gu23SLTT 曾因此错配到 Deng2021）。
            # 放宽到"过半命中"：PDF 抽字会丢词（Kag19 的标题词 "incrementally"
            # 在原文里跨行/成像，抽不出来），要求全中会把 Kag19 误判成"无匹配"。
            rare = [t for t in set(tk) if dfw.get(t, 0) <= 3]
            if rare:
                if sum(1 for t in rare if tok_hit(rec, t)) < len(rare) * 0.5:
                    continue
            # paper/ 的文件名本身就是"作者+年份"命名的，给同名同年文件加权，
            # 避免纯靠泛词匹配跑到别的 PDF 上（如 Lou20 配到 Bronstein）
            bm = re.match(r"^([A-Za-z'\-]+?)(\d{2,4})_",
                          os.path.splitext(fname)[0])
            boost = 0.0
            if bm and bm.group(1) == fam and bm.group(2) == item["year"]:
                boost = 0.25
            score = cov + boost
            if score > best_score + 1e-9:
                best, best_score, best_cov = fname, score, cov
    if best:
        return best, f"首作者[{first_fam}]命中 + 加权标题覆盖率{best_cov:.2f}"
    return None, ""


def audit(tex_path, paper_dir):
    tex = open(tex_path, encoding="utf-8").read()
    items = parse_bibitems(tex)
    index = build_paper_index(paper_dir)
    dfw = build_df(index, vocab(tex))
    print(f"解析到 {len(items)} 条 bibitem；paper/ 下 {len(index)} 个 PDF\n")

    results = []
    for it in items:
        authors, title, year = it["authors"], it["title"], it["year"]
        fname, how = match_pdf(it, index, dfw)
        problems = []
        rec = index.get(fname) if fname else None

        if rec is None:
            verdict = "无对应PDF"
            problems.append("paper/ 中没有任何 PDF 与其标题匹配")
        else:
            low = rec["lower"]
            # --- 作者核验：每个列出的作者姓氏是否真出现在 PDF 原文 ---
            missing = []
            for a in authors:
                # APA 格式 "Lopez de Compana, R. L." 里**姓氏是最后一段**。
                # 取第一段会只验 "Lopez"，把拼错的 "Compana" 放过去；
                # 取最后一段才能抓到这类拼写错误。
                fam = a.split()[-1] if a.split() else a
                if len(norm_name(fam)) < 3:
                    continue          # 跳过像 "M." 这类中间名缩写
                if not authored_by(rec, fam, zone_chars=AUTHOR_ZONE):
                    missing.append(a)
            if missing:
                problems.append("PDF 原文中查不到的作者: " + ", ".join(missing))
            # --- 标题核验：标题实词 token 在 PDF 原文中的覆盖率 ---
            tk = tokens(title)
            if tk:
                cov = sum(1 for t in tk if t in low) / len(tk)
                if cov < 0.55:
                    problems.append(f"标题 token 在 PDF 中覆盖率仅 {cov:.2f}")
        verdict = "OK" if not problems else "疑似幻觉"

        results.append(dict(
            key=it["key"], body=it["body"][:90], authors=authors, title=title,
            year=year, pdf=fname, how=how, verdict=verdict,
            problems="; ".join(problems),
            pdf_arxiv=rec["arxiv_id"] if rec else "",
            pdf_title=rec["title"][:80] if rec else "",
        ))
    return results, index


def render(results, index):
    print("=" * 100)
    print(f"{'判定':<10}{'bib key':<20}{'对应PDF':<48}说明")
    print("=" * 100)
    flags = 0
    for r in results:
        if r["verdict"] != "OK":
            flags += 1
        print(f"[{r['verdict']:<8}] {r['key']:<20} "
              f"{(r['pdf'] or '(无)')[:46]:<48}{r['problems'][:60]}")
    print("=" * 100)
    print(f"共 {len(results)} 条；异常 {flags} 条")
    print()
    print("--- 疑似异常详情 ---")
    for r in results:
        if r["verdict"] == "OK":
            continue
        print(f"* {r['key']}  ←→ PDF: {r['pdf'] or '(无匹配)'}")
        print(f"    引用: {r['body']}")
        print(f"    PDF实际: arXiv={r['pdf_arxiv']}  标题首行: {r['pdf_title']}")
        print(f"    问题: {r['problems']}")
        print()
    print("--- paper/ 文件名 vs PDF 内容一致性 ---")
    n = 0
    for fname, rec in sorted(index.items()):
        if rec.get("name_issue"):
            n += 1
            print(f"[装错内容] {fname}\n           → {rec['name_issue']}")
    if not n:
        print("(全部一致)")
    print()

    return flags


# 脚本自动判定的"疑似"仍需人工定性。以下结论已逐条回到 PDF 原文核对，
# 随报告一起输出，避免重跑后只剩一堆无解释的告警。
CURATED = {
    "Eshraghian23": ("确认错误",
                     "PDF 首页署名依次为 Jason K. Eshraghian、Max Ward、Emre Neftci、"
                     "Xinxin Wang、Gregor Lenz、**Girish Dwivedi**、Mohammed Bennamoun、"
                     "Doo Seok Jeong、Wei D. Lu。bibitem 写的 \"Dwiel, G.\" 在原文中不存在，"
                     "对应位置应为 Dwivedi。",
                     "改为 Dwivedi, G."),
    "MUTAG": ("确认错误",
              "PDF 首页署名：Asim Kumar Debnath, **Rosa L. Lopez de Compadre**, "
              "Gargi Debnath, Alan J. Shusterman, Corwin Hansch。bibitem 的 "
              "\"Lopez de Compana\" 是拼写错误。",
              "改为 Lopez de Compadre, R. L."),
    "Sun24MSG": ("确认错误",
                 "arXiv:2410.17941 首页署名：Li Sun、Zhenhao Huang、**Qiqi Wan**、"
                 "Hao Peng、Philip S. Yu。bibitem 列出的 \"Wang, Y.\" 与 \"Liu, C.\" "
                 "在这篇论文作者表中均不存在；第 3 作者是 Qiqi Wan，也未被列入。",
                 "删除 Wang, Y. 与 Liu, C.；Qiqi Wan 若需补则加在 Huang 后"),
    "Gu23SLTT": ("无法证实",
                 "paper/ 中不存在作者为 Gu/Deng/Li/Ding/Xie 的 SLTT 论文。"
                 "本地只有 Meng, Xiao, Yan, Wang, Lin, Luo 的 SLTT（arXiv 未标注，"
                 "正文标题为 Spatial Learning Through Time）。"
                 "引用的作者与出处均对不上。",
                 "按本地真实文献重写该条，或删除后改引 SLTT 原始作者"),
    "Ganev19": ("无法证实",
                "标题 \"Riemannian adaptive optimization methods\" 在 paper/ 中对应的是 "
                "Bécigneul & Ganea（ICLR 2019，arXiv:1810.00760）；"
                "bibitem 归属的 Ganev / Barp / Doucet 在任何本地 PDF 中都查不到，"
                "出处也写成 NeurIPS 2019。",
                "核实是否存在该 authored 版本；若不存在，改引 Bécigneul & Ganea"),
    "Roy19": ("本地无此文献",
              "paper/ 中没有任何 PDF 与 Roy, Jaiswal, Panda (2019) Nature 575:607-617 "
              "相符。注意 Deng2021 正文确实出现过 \"(Roy et al., 2019)\"，"
              "但那是该文自己的引用，不是本条文献。",
              "补下载原文，或改引本地已有的 SNN 综述"),
    "Lin23": ("本地无此文献",
              "paper/ 中没有 \"Hyperbolic representation learning for spiking neural "
              "networks\" 对应文献；脚本曾把它匹配到 SLTT 论文，但 SLTT 作者里没有 Xie。",
              "核实该文是否存在；若存在需补下载"),
}


def main():
    args = sys.argv[1:]
    tex_path = None
    paper_dir = r"D:/spikergnn/DvM/paper"
    out_path = None
    rest = list(args)
    if rest and not rest[0].startswith("--"):
        tex_path = rest.pop(0)
    if "--paper-dir" in rest:
        i = rest.index("--paper-dir")
        paper_dir = rest[i + 1]
    if "--out" in rest:
        out_path = rest[rest.index("--out") + 1]
    if not tex_path:
        here = os.path.dirname(os.path.abspath(__file__))
        tex_path = os.path.join(here, "post", "DvM_Paper_Anonymous.tex")

    results, index = audit(tex_path, paper_dir)
    flags = render(results, index)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as out:
            out.write("# 引用内容审计报告（PDF 原文基准）\n\n")
            out.write(f"基准来源：`DvM/paper/` 下 {len(index)} 篇真实 PDF 的原文文本。\n\n")
            out.write("## 汇总\n\n")
            out.write(f"- 待审计 bibitem：{len(results)} 条\n")
            out.write(f"- 异常：**{flags}** 条\n\n")
            out.write("## paper/ 文件名 vs 内容一致性\n\n")
            bad = [(f, r.get("name_issue", "")) for f, r in index.items() if r.get("name_issue")]
            if bad:
                for fname, iss in sorted(bad):
                    out.write(f"- **[装错内容]** `{fname}` — {iss}\n")
            else:
                out.write("(全部一致)\n")
            out.write("\n## 逐条明细\n\n")
            for r in results:
                out.write(f"### `{r['key']}` — {r['verdict']}\n\n")
                out.write(f"- 引用内容：{r['body']}\n")
                out.write(f"- 对应 PDF：`{r['pdf'] or '(无匹配)'}`"
                          f"（匹配依据：{r['how']}）\n")
                if r["pdf_arxiv"]:
                    out.write(f"- PDF 封面 arXiv 编号：{r['pdf_arxiv']}\n")
                if r["pdf_title"]:
                    out.write(f"- PDF 首页标题首行：{r['pdf_title']}\n")
                if r["problems"]:
                    out.write(f"- 问题：{r['problems']}\n")
                cur = CURATED.get(r["key"])
                if cur:
                    out.write(f"- 人工复核（{cur[0]}）：{cur[1]}\n")
                    out.write(f"- 处理建议：{cur[2]}\n")
                out.write("\n")
            out.write("## 判定口径\n\n")
            out.write("- 作者核验只在**正文前 6000 字符的署名区**进行，并排除引用语境"
                      "（`(... , Roy et al., 2019)`、`Han & Roy`），"
                      "避免被参考文献里的同名引用误导。\n")
            out.write("- 标题比对按**文档频次加权**：泛词（spiking/neural/network）"
                      "权重趋近 0，专名（SLTT/neuromorphic/trajectories）权重高。\n")
            out.write("- 命中的最低门槛：首作者姓氏命中 + 加权标题覆盖率 ≥ 0.45 + "
                      "稀有词（df≤3）过半命中。\n")
        print(f"\n报告已写入: {out_path}")
    return flags


if __name__ == "__main__":
    sys.exit(main())
