#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
引用内容审计脚本 (Citation Content Audit)
================================================
对论文 thebibliography 中每条 \\bibitem，自动抓取 arXiv / Crossref 元数据，
逐字段对比「标题 / 第一作者 / 年份」，输出"疑似幻觉引用"清单。

与"形式核验"(key 闭环 / DOI 可解析 / 可编译) 不同，本脚本做"内容核验"：
幻觉引用的特征是"格式正确、内容错误"，只有逐字段对比真实元数据才能抓到。

用法:
    python audit_citations.py [tex路径] [--out 报告路径]
默认 tex: post/DvM_Paper_Anonymous.tex
"""
import re, sys, time, json, html
import urllib.request, urllib.parse

HEADERS = {
    "User-Agent": "citation-audit/1.0 (mailto:audit@example.org)",
    "Accept": "application/atom+xml,application/xml,text/xml,application/json,*/*;q=0.8",
}


def fetch(url):
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise last_err


def clean_latex(s):
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\v\{c\}", "c", s)
    s = re.sub(r"\\'e", "e", s)
    s = re.sub(r"\\`o", "o", s)
    s = re.sub(r'\\"o', "o", s)
    s = re.sub(r"\\c\{c\}", "c", s)
    s = re.sub(r"\\&", "&", s)
    s = re.sub(r"\\_", "_", s)
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def norm_tokens(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return set(x for x in s.split() if len(x) > 1 or x.isdigit())


def parse_bibitems(tex):
    m = re.search(r"\\begin\{thebibliography\}.*?(?=\\section\{Reproducibility|\Z)", tex, re.S)
    if not m:
        return []
    block = m.group(0)
    items = []
    for bm in re.finditer(r"\\bibitem\[([^\]]*)\]\{([^}]*)\}(.*?)(?=\\bibitem\[|\Z)", block, re.S):
        label, key, raw = bm.groups()
        body = clean_latex(raw)
        ym = re.search(r"\((\d{4})\)", body)
        year = ym.group(1) if ym else "?"
        arm = re.search(r"arXiv[: ]*([0-9]{4}\.[0-9]{4,5})", body, re.I)
        arxiv = arm.group(1) if arm else ""
        dm = re.search(r"(10\.\d{4,9}/[^\s,;\)]+)", body)
        doi = dm.group(1).rstrip(".") if dm else ""
        # 第一作者姓氏 = 开头到第一个逗号
        first_author = body.split(",")[0].strip()
        items.append(dict(key=key, label=label, year=year, arxiv=arxiv,
                          doi=doi, first_author=first_author, body=body))
    return items


# ---------- 真实元数据查询 ----------

def q_arxiv_by_id(arxiv):
    url = f"https://export.arxiv.org/api/query?id_list={arxiv}"
    xml = fetch(url)
    title = re.search(r"<title>([^<]+)</title>", xml)
    authors = re.findall(r"<name>([^<]+)</name>", xml)
    pub = re.search(r"<published>(\d{4})", xml)
    return {
        "title": html.unescape(title.group(1).strip()) if title else "",
        "authors": authors,
        "year": pub.group(1) if pub else "",
    }


def q_arxiv_by_title(title):
    q = urllib.parse.quote(title)
    url = f"https://export.arxiv.org/api/query?search_query=ti:%22{q}%22&max_results=3"
    xml = fetch(url)
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
    if not entries:
        return None
    e = entries[0]
    t = re.search(r"<title>([^<]+)</title>", e)
    authors = re.findall(r"<name>([^<]+)</name>", e)
    pub = re.search(r"<published>(\d{4})", e)
    return {
        "title": html.unescape(t.group(1).strip()) if t else "",
        "authors": authors,
        "year": pub.group(1) if pub else "",
    }


def q_crossref_by_doi(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    try:
        data = json.loads(fetch(url))
        msg = data["message"]
        title = (msg.get("title") or [""])[0]
        authors = [(a.get("family", "") + " " + a.get("given", "")).strip()
                   for a in msg.get("author", [])]
        year = ""
        for k in ("published-print", "published-online", "issued", "created"):
            if k in msg and msg[k].get("date-parts"):
                year = str(msg[k]["date-parts"][0][0])
                break
        return {"title": title, "authors": authors, "year": year}
    except Exception as e:
        return {"title": "", "authors": [], "year": "", "error": str(e)}


def q_crossref_by_title(title):
    url = "https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(title)
    try:
        data = json.loads(fetch(url))
        items = data["message"]["items"]
        if not items:
            return None
        it = items[0]
        t = (it.get("title") or [""])[0]
        authors = [(a.get("family", "") + " " + a.get("given", "")).strip()
                   for a in it.get("author", [])]
        year = ""
        for k in ("published-print", "published-online", "issued", "created"):
            if k in it and it[k].get("date-parts"):
                year = str(it[k]["date-parts"][0][0])
                break
        return {"title": t, "authors": authors, "year": year}
    except Exception:
        return None


# ---------- 对比 ----------

def first_family(name):
    parts = name.replace("&", " ").split()
    return parts[0] if parts else ""


def family_norm(s):
    s = s.lower().replace("'", "").replace("`", "")
    s = re.sub(r"[^a-z]+", "", s)
    return s


def title_overlap(cited_body, real_title):
    """真实标题的 token 在 bibitem 正文中的覆盖率 (0~1)"""
    A = norm_tokens(real_title)
    B = norm_tokens(cited_body)
    if not A:
        return 1.0
    return len(A & B) / len(A)


def audit(tex_path):
    tex = open(tex_path, encoding="utf-8").read()
    items = parse_bibitems(tex)
    print(f"解析到 {len(items)} 条 bibitem\n")

    report = []
    for it in items:
        key = it["key"]
        real = None
        src = ""
        try:
            # 1) 优先 DOI
            if it["doi"]:
                real = q_crossref_by_doi(it["doi"])
                src = f"DOI:{it['doi']}"
            # 2) 其次 arXiv 编号
            elif it["arxiv"]:
                real = q_arxiv_by_id(it["arxiv"])
                src = f"arXiv:{it['arxiv']}"
            # 3) 无编号：标题搜索（arXiv → Crossref 兜底）
            else:
                t = re.sub(r"^.*?\(\d{4}\)\.\s*", "", it["body"])
                t = re.split(r"\.\s+(?=Paper|In \b|https|\[Computer)", t)[0].strip(" .")
                real = q_arxiv_by_title(t)
                src = f"title-search(arXiv):{t[:60]}"
                if real is None:
                    real = q_crossref_by_title(t)
                    src = f"title-search(Crossref):{t[:60]}"
        except Exception as e:
            real = {"title": "", "authors": [], "year": "", "error": str(e)}
            src = src + f" [查询异常:{type(e).__name__}]"

        if real is None or not real.get("title"):
            report.append((key, "未查到", "", "", "", it["body"][:80]))
            continue

        # 逐字段对比
        real_title = real["title"]
        real_first = first_family(real["authors"][0]) if real["authors"] else ""
        real_year = real.get("year", "")

        ov = title_overlap(it["body"], real_title)
        cited_first = first_family(it["first_author"])
        author_ok = family_norm(cited_first) == family_norm(real_first)
        year_ok = (real_year == "") or (real_year == it["year"])

        verdict = "OK"
        problems = []
        if ov < 0.55:
            verdict = "疑似幻觉"
            problems.append(f"标题相似度仅 {ov:.2f}")
        elif ov < 0.75:
            problems.append(f"标题相似度偏低 {ov:.2f}")
        if not author_ok:
            problems.append(f"第一作者不符: 引用[{cited_first}] vs 真实[{real_first}]")
            if verdict == "OK":
                verdict = "作者不符"
        if not year_ok:
            problems.append(f"年份不符: 引用[{it['year']}] vs 真实[{real_year}]")

        report.append((key, verdict, real_title, real_first, real_year,
                       it["body"][:70], "; ".join(problems), ov, src))

    return report


def render(report):
    print("=" * 100)
    print("引用内容审计结果")
    print("=" * 100)
    flags = 0
    for r in report:
        if len(r) == 6:
            key, verdict, _, _, _, body = r
            print(f"[未查到] {key:20s}  {body}")
            continue
        key, verdict, real_title, real_first, real_year, body, problems, ov, src = r
        mark = {"OK": "  [OK]  ", "疑似幻觉": "[幻觉] ", "作者不符": "[作者] "}.get(verdict, "  [?]  ")
        if verdict != "OK":
            flags += 1
        print(f"{mark}{key:18s} 真实: {real_title[:70]}")
        print(f"           引用: {body}")
        print(f"           第一作者[{real_first}] 年份[{real_year}] 相似度[{ov:.2f}] 来源[{src}]")
        if problems:
            print(f"           → {problems}")
        print()
    print("=" * 100)
    print(f"共 {len(report)} 条；异常 {flags} 条（疑似幻觉 / 作者不符 / 未查到）")
    return flags


def main():
    tex_path = "post/DvM_Paper_Anonymous.tex"
    out_path = None
    args = sys.argv[1:]
    if args and not args[0].startswith("--"):
        tex_path = args[0]
        args = args[1:]
    if "--out" in args:
        out_path = args[args.index("--out") + 1]

    report = audit(tex_path)
    flags = render(report)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# 引用内容审计报告\n\n")
            for r in report:
                if len(r) == 6:
                    key, verdict, _, _, _, body = r
                    f.write(f"- **[未查到]** `{key}` — {body}\n")
                    continue
                key, verdict, real_title, real_first, real_year, body, problems, ov, src = r
                f.write(f"- **[{verdict}]** `{key}` — 真实标题: {real_title}; "
                        f"第一作者: {real_first}; 年份: {real_year}; 相似度: {ov:.2f}\n")
                if problems:
                    f.write(f"  - 问题: {problems}\n")
        print(f"\n报告已写入: {out_path}")
    return flags


if __name__ == "__main__":
    sys.exit(main())
