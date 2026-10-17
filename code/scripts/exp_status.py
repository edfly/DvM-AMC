"""exp_status.py -- one-shot status/ETA probe for the running experiment queue.

Answers, in a single call: is the process alive, which job is running, how far
along it is, what the wall-clock ETA of the WHOLE queue is, and has the machine
been restarted since the queue started (the failure mode that silently cost us
a job on 2026-09-28).

Usage
    python exp_status.py                # full report
    python exp_status.py --json         # machine-readable

Design notes
    * The queue is rerun_fixed_supplement.py: 4 datasets x 3 targets
      (OM 5-shot, No-Spikes 1-shot, No-Spikes 5-shot) = 12 jobs, run in that
      order by a single driver process.
    * Per-job cost is calibrated from MEASURED runs, not guessed: every job is
      9 runs x 100 epochs, so the wall clock is dominated by the fixed
      schedule.  Measured totals: MUTAG 1-shot 41684 s, MUTAG OM 5-shot
      39740 s, PROTEINS OM 5-shot 38318 s, PROTEINS 39382 s, NCI1 42863 s,
      DD 41856 s -> ~11.4 h/job with <10 % spread across datasets (an earlier
      "DD is 3x slower" guess was wrong and would have overstated the queue).
    * The currently running job is measured live from the tqdm line of its
      log, so the ETA tightens as the job progresses.
"""

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "rerun_outputs")
LOGS = os.path.join(ROOT, "rerun_logs")

DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD"]
# (label, shot, subdir, stem)  -- MUST mirror rerun_fixed_supplement._TARGETS
# Order matches the driver, so "next job" predictions follow the real queue.
# NOTE 2026-10-03: the OM 5-shot row was missing here, which under-counted the
# queue by 4 jobs (~1.9 days) -> ETA was reported ~2 days too early.  Keep this
# list identical to _TARGETS in rerun_fixed_supplement.py (12 jobs total).
TARGETS = [("OM", "5", "OM", "5shot_full"),
           ("No-Spikes", "1", "NoSpikesFixed", "1shot_no_spikes"),
           ("No-Spikes", "5", "NoSpikesFixed", "5shot_no_spikes")]

JOB_HOURS = 11.4          # measured mean per job (9 runs x 100 epochs)
TRAIN_MIN_PER_RUN = 0.0   # filled from the live tqdm line

PROGRESS = re.compile(
    r"(?P<label>[\w\-]+)\s+(?P<shot>\d)-shot\s+run(?P<run>\d+):\s*"
    r"(?P<pct>\d+)%.*?\[(?P<el>\d+):(?P<em>\d+)<(?P<ra>\d+):(?P<rm>\d+),"
    r"\s*(?P<sit>[\d.]+)s/it")


def _tail(path, n=4000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "ignore")
    except OSError:
        return ""


def _procs(min_mb=30):
    """Count RESIDENT python processes (exact image match).

    Transient launchers show up in tasklist for a fraction of a second with a
    tiny working set (e.g. 21 MB) and inflated the count to 3 while only 2
    real processes existed; the size filter removes that noise.
    """
    try:
        r = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True,
                           text=True, encoding="gbk", errors="ignore", timeout=30)
        n = 0
        for line in r.stdout.splitlines():
            cols = [c.strip('"') for c in line.split('","')]
            if len(cols) < 5:
                continue
            if cols[0].lower() not in ("python.exe", "pythonw.exe"):
                continue
            try:
                kb = float(cols[4].replace(" K", "").replace(",", "").strip())
            except ValueError:
                continue
            if kb >= min_mb * 1024:
                n += 1
        return n
    except Exception:
        return -1


def _last_boot():
    """Boot time via GetTickCount64 (no PowerShell, no locale pitfalls)."""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.GetTickCount64.restype = ctypes.c_ulonglong
        up_ms = int(k32.GetTickCount64())
        boot = dt.datetime.now() - dt.timedelta(milliseconds=up_ms)
        return boot.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"


def _fmt_h(h):
    if h is None or h != h:
        return "?"
    total = int(round(h * 60))
    d, rem = divmod(total, 24 * 60)
    hh, mm = divmod(rem, 60)
    return f"{d}d {hh}h {mm:02d}m" if d else f"{hh}h {mm:02d}m"


DRIVER = "rerun_fixed_supplement.py"


def check_targets_sync():
    """Assert this view's job table equals the driver's (AST, no side effects).

    Written because the OM 5-shot row was once missing here, silently
    under-reporting the queue by 4 jobs (~1.9 days of ETA).
    """
    path = os.path.join(ROOT, DRIVER)
    if not os.path.exists(path):
        return None, f"driver not found: {DRIVER}"
    try:
        import ast
        tree = ast.parse(open(path, encoding="utf-8").read())
        driver = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "_TARGETS":
                        driver = [tuple(ast.literal_eval(e)) for e in node.value.elts]
        if driver is None:
            return None, f"_TARGETS not found in {DRIVER}"
    except Exception as e:  # noqa
        return None, f"parse failed: {e}"
    # driver rows are (method_label, --ablation, shot, save_subdir, out_stem);
    # this view keeps (method_label, shot, save_subdir, out_stem) -> project.
    proj = [t if len(t) == 4 else (t[0], t[2], t[3], t[4]) for t in driver]
    mine = [tuple(x) for x in TARGETS]
    if mine != proj:
        return False, (f"TARGETS out of sync with {DRIVER}:\n"
                       f"      here   = {mine}\n"
                       f"      driver = {proj}")
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows, pending, running = [], [], None
    now_dt = dt.datetime.now()
    sync_ok, sync_msg = check_targets_sync()
    for ds in DATASETS:
        for label, shot, sub, stem in TARGETS:
            j = os.path.join(OUT, f"{ds}_{sub}", f"{ds}_{stem}.json")
            done = os.path.exists(j)
            log = os.path.join(LOGS, f"{ds}__{label.replace('-', '-')}__{shot}shot.log")
            r = {"dataset": ds, "label": label, "shot": shot,
                 "done": done, "json": j}
            if not done:
                pending.append(r)
                m = PROGRESS.search(_tail(log)) if os.path.exists(log) else None
                if m:
                    run = int(m.group("run"))
                    pct = int(m.group("pct"))
                    sit = float(m.group("sit"))
                    left_h = ((9 - run) + (100 - pct) / 100.0) * 100 * sit / 3600.0
                    r["progress"] = (f"run{run}/9 {pct:>3d}% "
                                     f"@ {sit:.0f}s/it")
                    r["eta_h"] = round(left_h, 2)
                    r["log_age_min"] = round(
                        (now_dt - dt.datetime.fromtimestamp(
                            os.path.getmtime(log))).total_seconds() / 60.0, 1)
                    running = r
            rows.append(r)

    n_pending = len(pending)
    total_left = 0.0
    for r in pending:
        total_left += r.get("eta_h", JOB_HOURS)
    now = now_dt
    finish = now + dt.timedelta(hours=total_left)

    rep = {
        "checked_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "last_boot": _last_boot(),
        "python_procs": _procs(),
        "jobs_total": len(rows),
        "jobs_done": len(rows) - n_pending,
        "jobs_pending": n_pending,
        "targets_in_sync": sync_ok,
        "targets_note": sync_msg,
        "running": running,
        "queue_eta_hours": round(total_left, 1),
        "queue_eta_human": _fmt_h(total_left),
        "projected_finish": finish.strftime("%Y-%m-%d %H:%M"),
        "per_job_remaining": pending,
    }

    if args.json:
        print(json.dumps(rep, indent=1, ensure_ascii=False))
        return

    print("=" * 66)
    print(f"实验队列状态   {rep['checked_at']}")
    print("=" * 66)
    print(f"  任务总数        {rep['jobs_total']}  "
          f"(完成 {rep['jobs_done']} / 待跑 {rep['jobs_pending']})")
    print(f"  python 进程数   {rep['python_procs']}")
    print(f"  机器最近启动    {rep['last_boot']}")
    if running:
        print(f"  当前 job        {running['dataset']} / {running['label']} / "
              f"{running['shot']}-shot  [{running['progress']}]")
        age = running.get("log_age_min", 0.0)
        flag = "OK" if age < 30 else "!! 日志停滞，可能被杀/重启"
        print(f"  日志新鲜度      {age:.1f} min  [{flag}]")
    print(f"  单 job 基准     {JOB_HOURS} h（实测均值，9 run x 100 epoch）")
    if sync_ok is False:
        print("  ⚠️ 口径不一致  TARGETS 与驱动脚本不同步，下表可能漏 job：")
        print(f"      {sync_msg}")
    elif sync_ok is None:
        print(f"  (口径自检跳过：{sync_msg})")
    done_rows = [r for r in rows if r["done"]]
    if done_rows:
        print("-" * 66)
        print(f"  已完成 {len(done_rows)} 个：")
        for r in done_rows:
            ts = dt.datetime.fromtimestamp(os.path.getmtime(r["json"]))
            print(f"    {r['dataset']:>8} / {r['label']:<9} / {r['shot']}-shot   "
                  f"{ts.strftime('%m-%d %H:%M')}")
    print("-" * 66)
    print("  待跑 job 与估算：")
    for r in pending:
        e = r.get("eta_h")
        print(f"    {r['dataset']:>8} / {r['label']:<9} / {r['shot']}-shot   "
              f"{_fmt_h(e) if e else _fmt_h(JOB_HOURS):>10}"
              f"   {'' if e else '(按基准)'}")
    print("-" * 66)
    print(f"  队列总剩余      {rep['queue_eta_human']}")
    print(f"  预计全部完成    {rep['projected_finish']}")
    print("  （完成后仍需人工：重跑 summarize_matrix.py + recompute_energy.py"
          " + 论文回填，约 0.5–1 天）")


if __name__ == "__main__":
    main()
