"""
EC2 Instance Family Micro-Benchmark (rows/s)

목적
- 인스턴스 패밀리/아키텍처(x86 vs ARM)에 따른 성능 차이를 비교
- CPU 연산(해시), 파일 쓰기(버퍼드/선택적 fsync), SQLite insert(PRAGMA 튜닝) 측정

출력
- 각 테스트별 median/p95/min/max 시간 + throughput(rows/s)
"""

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
# 0) SETTINGS
# ==========================================================
CSV_PATH = "sap500_data.csv"
OUTDIR   = "."

LIMIT_ROWS = 0
REPEATS = 5                   # 반복 횟수
WORKERS = "auto"

# --- CPU 테스트 강도 (해시 반복 횟수)
CPU_ROUNDS = 30

# --- 파일 쓰기 테스트 설정
BLOCK_LINES = 5000            # buffered write block 크기
FSYNC_LINES = 0               

# --- SQLite insert 테스트 설정
SQLITE_COMMIT_EVERY = 1000    # 커밋 단위
# ==========================================================


# ==========================================================
# 1) 유틸리티 (시간/통계/시스템 정보)
# ==========================================================
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
# ==========================================================


# ==========================================================
# 2) 데이터 로딩 (모든 테스트가 동일 데이터 사용)
# ==========================================================
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
    lines: List[bytes] = []
    lines.append((",".join(data.header) + "\n").encode("utf-8"))
    for r in data.rows:
        lines.append((",".join(r) + "\n").encode("utf-8"))
    return lines
# ==========================================================


# ==========================================================
# 3) TEST 1 : CPU-bound
#    - multiprocessing으로 코어를 최대한 활용
# ==========================================================
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
    chunks = [rows[i:i + chunk_size] for i in range(0, n, chunk_size)]

    start = now_perf()
    if w == 1:
        _hash_worker((rows, rounds))
    else:
        with Pool(processes=w) as pool:
            pool.map(_hash_worker, [(c, rounds) for c in chunks])
    return now_perf() - start
# ==========================================================


# ==========================================================
# 4) TEST 2: I/O write
# ==========================================================
def bench_io_write(lines: List[bytes], out_path: str, fsync_every: int, block_lines: int) -> float:
    start = now_perf()
    with open(out_path, "wb", buffering=1024 * 1024) as f:
        buf: List[bytes] = []
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
# ==========================================================


# ==========================================================
# 5) TEST 3: SQLite insert
# ==========================================================
def bench_sqlite_insert_pragmas(
    rows: List[List[str]],
    header: List[str],
    db_path: str,
    commit_every: int,
) -> float:
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")   # ~200MB
    cur.execute("PRAGMA mmap_size=268435456;")  # 256MB

    # 테이블 생성 (CSV 컬럼을 전부 TEXT로)
    col_def = ", ".join([f'"{c}" TEXT' for c in header])
    cur.execute(f"CREATE TABLE test_table ({col_def})")

    # INSERT 준비
    placeholders = ",".join(["?"] * len(header))
    q = f"INSERT INTO test_table VALUES ({placeholders})"

    # INSERT + 주기적 commit
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
# ==========================================================


# ==========================================================
# 6) Main
# ==========================================================
def main():
    # (A) 입력/출력 경로 확인
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV_PATH not found: {CSV_PATH} (경로를 하드코딩 값으로 수정하세요)")

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # (B) 실행 파라미터 결정
    workers = cpu_count() if WORKERS == "auto" else max(1, int(WORKERS))
    limit = None if LIMIT_ROWS == 0 else LIMIT_ROWS

    # (C) 헤더 출력(재현성 기록)
    print("=" * 72)
    print("System:", sys_info())
    print("CSV_PATH:", CSV_PATH)
    print("OUTDIR:", str(outdir.resolve()))
    print(f"workers={workers}, repeats={REPEATS}, cpu_rounds={CPU_ROUNDS}")
    print(f"block_lines={BLOCK_LINES}, fsync_lines={FSYNC_LINES}, sqlite_commit_every={SQLITE_COMMIT_EVERY}")
    print("=" * 72)

    # (D) 데이터 로딩 (모든 테스트가 동일 데이터 사용)
    data = load_csv_data(CSV_PATH, limit=limit)
    nrows = len(data.rows)
    print(f"[*] Loaded rows: {nrows:,}, cols: {len(data.header)}")

    # 파일쓰기용 bytes 라인 생성(1회)
    lines = build_csv_lines_bytes(data)

    # ------------------------------------------------------
    # (1) CPU_HASH
    # ------------------------------------------------------
    cpu_times = []
    for _ in range(REPEATS):
        cpu_times.append(bench_cpu_hash(data.rows, rounds=CPU_ROUNDS, workers=workers))
    cpu_thr = nrows / statistics.median(cpu_times)

    # ------------------------------------------------------
    # (2) IO_BUFFERED_WRITE
    # ------------------------------------------------------
    io_buf_times = []
    csv_buf_path = str(outdir / "bench_buffered.csv")
    for _ in range(REPEATS):
        io_buf_times.append(bench_io_write(lines, csv_buf_path, fsync_every=0, block_lines=BLOCK_LINES))
    io_buf_thr = nrows / statistics.median(io_buf_times)

    # ------------------------------------------------------
    # (3) IO_FSYNC_WRITE
    # ------------------------------------------------------
    io_fsync_times = []
    io_fsync_thr = None
    if FSYNC_LINES > 0:
        csv_fsync_path = str(outdir / "bench_fsync.csv")
        for _ in range(REPEATS):
            io_fsync_times.append(bench_io_write(lines, csv_fsync_path, fsync_every=FSYNC_LINES, block_lines=1))
        io_fsync_thr = nrows / statistics.median(io_fsync_times)

    # ------------------------------------------------------
    # (4) SQLITE_PRAGMAS
    # ------------------------------------------------------
    sqlite_pragmas_times = []
    db_pragmas_path = str(outdir / "bench_pragmas.db")
    for _ in range(REPEATS):
        sqlite_pragmas_times.append(
            bench_sqlite_insert_pragmas(data.rows, data.header, db_pragmas_path, commit_every=SQLITE_COMMIT_EVERY)
        )
    sqlite_pragmas_thr = nrows / statistics.median(sqlite_pragmas_times)

    # (E) 결과 출력
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

    print(f"\n[SQLITE_PRAGMAS (commit_every={SQLITE_COMMIT_EVERY})]")
    print("  " + summarize_times(sqlite_pragmas_times))
    print(f"  throughput: {sqlite_pragmas_thr:,.2f} rows/s")


if __name__ == "__main__":
    main()
