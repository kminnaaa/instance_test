import csv
import hashlib
import os
import platform
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional

# ==========================================================
# ✅ HARD-CODED SETTINGS (여기만 바꾸면 됨)
# ==========================================================
CSV_PATH = "sap500_data.csv"   # 예: "/home/ubuntu/sap500_data.csv"
OUTDIR   = "."                # 예: "/tmp" (EBS), "/mnt/nvme" (NVMe)

LIMIT_ROWS = 0                # 0이면 전체, 예: 5000
REPEATS = 5                   # 반복 횟수 (median/p95 안정화)
WORKERS = "auto"              # "auto" 또는 숫자 (예: 4)
CPU_ROUNDS = 30               # CPU 차이 크게 보려면 ↑ (예: 30~80)

BLOCK_LINES = 5000            # buffered write block 크기(throughput)
FSYNC_LINES = 0               # fsync 매 N줄 (0이면 비활성, 예: 1 또는 200)

SQLITE_COMMIT_EVERY = 1000    # 1=매행 커밋(느림), 1000 추천, 0=끝에서 1번만 커밋
# ==========================================================


# -----------------------------
# Utilities
# -----------------------------
def now_perf():
    return time.perf_counter()

def fmt_sec(x: float) -> str:
    return f"{x:.4f}s"

def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def sys_info():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
    }

def summarize_times(times: List[float]) -> str:
    med = statistics.median(times)
    p95 = percentile(times, 0.95)
    mn = min(times)
    mx = max(times)
    return f"median={fmt_sec(med)}, p95={fmt_sec(p95)}, min={fmt_sec(mn)}, max={fmt_sec(mx)}"


# -----------------------------
# Data loading (same data)
# -----------------------------
@dataclass
class CsvData:
    header: List[str]
    rows: List[List[str]]

def load_csv_data(file_path: str, limit: Optional[int] = None) -> CsvData:
    rows: List[List[str]] = []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, r in enumerate(reader):
            rows.append(r)
            if limit is not None and (i + 1) >= limit:
                break
    return CsvData(header=header, rows=rows)

def build_csv_lines_bytes(data: CsvData) -> List[bytes]:
    lines = []
    lines.append((",".join(data.header) + "\n").encode("utf-8"))
    for r in data.rows:
        lines.append((",".join(r) + "\n").encode("utf-8"))
    return lines


# -----------------------------
# Benchmarks
# -----------------------------
def _hash_worker(args):
    chunk_rows, rounds = args
    out = 0
    for _ in range(rounds):
        for r in chunk_rows:
            h = hashlib.sha256()
            for v in r:
                h.update(v.encode("utf-8", errors="ignore"))
            out ^= int.from_bytes(h.digest()[:8], "little")
    return out

def bench_cpu_hash(rows: List[List[str]], rounds: int, workers: int) -> float:
    n = len(rows)
    w = max(1, workers)
    chunk_size = (n + w - 1) // w
    chunks = [rows[i:i+chunk_size] for i in range(0, n, chunk_size)]

    start = now_perf()
    if w == 1:
        _hash_worker((rows, rounds))
    else:
        with Pool(processes=w) as pool:
            pool.map(_hash_worker, [(c, rounds) for c in chunks])
    return now_perf() - start

def bench_io_write(lines: List[bytes], out_path: str, fsync_every: int, block_lines: int) -> float:
    """
    fsync_every:
      - 0  : no fsync (buffered)
      - N>0: fsync every N lines (latency-sensitive)
    """
    start = now_perf()
    with open(out_path, "wb", buffering=1024 * 1024) as f:
        buf = []
        for i, line in enumerate(lines, start=1):
            buf.append(line)
            if len(buf) >= block_lines:
                f.write(b"".join(buf))
                buf.clear()

            if fsync_every > 0 and (i % fsync_every == 0):
                f.flush()
                os.fsync(f.fileno())

        if buf:
            f.write(b"".join(buf))

        f.flush()
        if fsync_every > 0:
            os.fsync(f.fileno())

    return now_perf() - start

def bench_sqlite_insert(
    rows: List[List[str]],
    header: List[str],
    db_path: str,
    commit_every: int,
    pragmas: bool
) -> float:
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if pragmas:
        # CPU 차이 더 잘 보이게 (I/O 지배 완화)
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA cache_size=-200000;")   # ~200MB
        cur.execute("PRAGMA mmap_size=268435456;")  # 256MB

    col_def = ", ".join([f'"{c}" TEXT' for c in header])
    cur.execute(f"CREATE TABLE test_table ({col_def})")

    placeholders = ",".join(["?"] * len(header))
    q = f"INSERT INTO test_table VALUES ({placeholders})"

    start = now_perf()
    pending = 0
    cur.execute("BEGIN;")
    for r in rows:
        cur.execute(q, r)
        pending += 1
        if commit_every > 0 and pending >= commit_every:
            conn.commit()
            cur.execute("BEGIN;")
            pending = 0

    conn.commit()
    conn.close()
    return now_perf() - start


# -----------------------------
# Main
# -----------------------------
def main():
    # Check paths
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV_PATH not found: {CSV_PATH} (경로를 하드코딩 값으로 수정하세요)")

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    workers = cpu_count() if WORKERS == "auto" else max(1, int(WORKERS))
    limit = None if LIMIT_ROWS == 0 else LIMIT_ROWS

    print("=" * 72)
    print("System:", sys_info())
    print("CSV_PATH:", CSV_PATH)
    print("OUTDIR:", str(outdir.resolve()))
    print(f"workers={workers}, repeats={REPEATS}, cpu_rounds={CPU_ROUNDS}")
    print(f"block_lines={BLOCK_LINES}, fsync_lines={FSYNC_LINES}, sqlite_commit_every={SQLITE_COMMIT_EVERY}")
    print("=" * 72)

    data = load_csv_data(CSV_PATH, limit=limit)
    nrows = len(data.rows)
    print(f"[*] Loaded rows: {nrows:,}, cols: {len(data.header)}")

    lines = build_csv_lines_bytes(data)

    # 1) CPU-bound
    cpu_times = []
    for _ in range(REPEATS):
        cpu_times.append(bench_cpu_hash(data.rows, rounds=CPU_ROUNDS, workers=workers))
    cpu_thr = nrows / statistics.median(cpu_times)

    # 2) I/O buffered throughput
    io_buf_times = []
    csv_buf_path = str(outdir / "bench_buffered.csv")
    for _ in range(REPEATS):
        io_buf_times.append(bench_io_write(lines, csv_buf_path, fsync_every=0, block_lines=BLOCK_LINES))
    io_buf_thr = nrows / statistics.median(io_buf_times)

    # 3) I/O fsync latency (optional)
    io_fsync_times = []
    io_fsync_thr = None
    if FSYNC_LINES > 0:
        csv_fsync_path = str(outdir / "bench_fsync.csv")
        for _ in range(REPEATS):
            io_fsync_times.append(bench_io_write(lines, csv_fsync_path, fsync_every=FSYNC_LINES, block_lines=1))
        io_fsync_thr = nrows / statistics.median(io_fsync_times)

    # 4) SQLite default
    sqlite_times = []
    db_path = str(outdir / "bench_default.db")
    for _ in range(REPEATS):
        sqlite_times.append(
            bench_sqlite_insert(data.rows, data.header, db_path, commit_every=SQLITE_COMMIT_EVERY, pragmas=False)
        )
    sqlite_thr = nrows / statistics.median(sqlite_times)

    # 5) SQLite pragmas
    sqlite_fast_times = []
    db_fast_path = str(outdir / "bench_pragmas.db")
    for _ in range(REPEATS):
        sqlite_fast_times.append(
            bench_sqlite_insert(data.rows, data.header, db_fast_path, commit_every=SQLITE_COMMIT_EVERY, pragmas=True)
        )
    sqlite_fast_thr = nrows / statistics.median(sqlite_fast_times)

    # Print results
    print("\n" + "=" * 72)
    print(f"RESULTS (rows={nrows:,})")
    print("=" * 72)

    print(f"\n[CPU_HASH (rounds={CPU_ROUNDS}, workers={workers})]")
    print("  " + summarize_times(cpu_times))
    print(f"  throughput: {cpu_thr:,.2f} rows/s")

    print(f"\n[IO_BUFFERED_WRITE (block_lines={BLOCK_LINES})]")
    print("  " + summarize_times(io_buf_times))
    print(f"  throughput: {io_buf_thr:,.2f} rows/s")

    if FSYNC_LINES > 0:
        print(f"\n[IO_FSYNC_WRITE (fsync_every={FSYNC_LINES} lines)]")
        print("  " + summarize_times(io_fsync_times))
        print(f"  throughput: {io_fsync_thr:,.2f} rows/s")

    print(f"\n[SQLITE_DEFAULT (commit_every={SQLITE_COMMIT_EVERY})]")
    print("  " + summarize_times(sqlite_times))
    print(f"  throughput: {sqlite_thr:,.2f} rows/s")

    print(f"\n[SQLITE_PRAGMAS (commit_every={SQLITE_COMMIT_EVERY})]")
    print("  " + summarize_times(sqlite_fast_times))
    print(f"  throughput: {sqlite_fast_thr:,.2f} rows/s")

    print("\nTips:")
    print(" - 패밀리(C vs M) 차이 크게: CPU_ROUNDS를 50~100으로 올리세요.")
    print(" - 스토리지(EBS vs NVMe) 차이: OUTDIR을 NVMe 마운트로 바꿔서 IO/SQLite 비교하세요.")
    print(" - commit_every=1은 다시 fsync 지배가 되어 패밀리 차이가 안 보이는 게 정상입니다.")
    print("=" * 72)


if __name__ == "__main__":
    main()
