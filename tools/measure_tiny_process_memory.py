#!/usr/bin/env python3
"""Sample a command's process-tree memory while leaving external watchdogs intact."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil


def proc_metric_bytes(pid: int, filename: str, metric: str) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/{filename}").read_text().splitlines():
            if line.startswith(f"{metric}:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError, ValueError):
        pass
    return None


def process_memory(process: psutil.Process) -> dict:
    try:
        memory = process.memory_full_info()
        try:
            cmdline = process.cmdline()[:3]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cmdline = []
        try:
            name = process.name()
            ppid = process.ppid()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            name, ppid = "<exited>", None
        rss_anon = proc_metric_bytes(process.pid, "status", "RssAnon")
        swap_pss = proc_metric_bytes(process.pid, "smaps_rollup", "SwapPss")
        return {
            "pid": process.pid,
            "ppid": ppid,
            "name": name,
            "cmdline": cmdline,
            "rss_bytes": memory.rss,
            "uss_bytes": getattr(memory, "uss", None),
            "pss_bytes": getattr(memory, "pss", None),
            "swap_bytes": getattr(memory, "swap", None),
            "rss_anon_bytes": rss_anon,
            "swap_pss_bytes": swap_pss,
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {"pid": process.pid, "name": "<exited>"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after artifact_dir (for example: DIR -- COMMAND ...)")

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(command)
    root_process = psutil.Process(child.pid)
    start = time.monotonic()
    output = args.artifact_dir / "process-memory.jsonl"
    with output.open("w") as stream:
        while child.poll() is None:
            try:
                tree = [root_process, *root_process.children(recursive=True)]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                tree = [root_process]
            record = {"elapsed": time.monotonic() - start, "processes": [process_memory(p) for p in tree]}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            time.sleep(2)
    return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
