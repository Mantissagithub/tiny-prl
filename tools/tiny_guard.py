"""Monitor a training process inside the launcher's memory-limited systemd scope."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def stop_reason(available, disk, gpu_mib, elapsed, timeout):
    if available < 2 * 1024**3:
        return "available host RAM below 2 GiB"
    if disk < 2 * 1024**3:
        return "free disk below 2 GiB"
    if gpu_mib > 7676:
        return "GPU memory exceeded 7676 MiB"
    if elapsed > timeout:
        return "run timeout"
    return None


def main():
    artifact = Path(sys.argv[1])
    timeout = int(sys.argv[2])
    command = sys.argv[3:]
    cg = next(
        line.split("::", 1)[1] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")
    )
    cgroup = Path("/sys/fs/cgroup") / cg.lstrip("/")
    interrupted = False

    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, on_signal)
    start = time.monotonic()
    minimum_ram = minimum_disk = None
    peak_memory = peak_swap = peak_gpu = 0
    reason = None
    proc = subprocess.Popen(command, start_new_session=True)
    try:
        with (artifact / "resources.jsonl").open("w") as log:
            while proc.poll() is None:
                available = next(
                    int(line.split()[1]) * 1024
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                )
                disk = shutil.disk_usage(artifact).free
                gpu = int(
                    subprocess.check_output(
                        ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        text=True,
                        timeout=5,
                    ).strip()
                )
                memory = int((cgroup / "memory.current").read_text())
                swap = int((cgroup / "memory.swap.current").read_text())
                elapsed = time.monotonic() - start
                minimum_ram = available if minimum_ram is None else min(minimum_ram, available)
                minimum_disk = disk if minimum_disk is None else min(minimum_disk, disk)
                peak_memory, peak_swap, peak_gpu = max(peak_memory, memory), max(peak_swap, swap), max(peak_gpu, gpu)
                log.write(
                    json.dumps(
                        {
                            "elapsed_seconds": elapsed,
                            "available_host_bytes": available,
                            "disk_free_bytes": disk,
                            "cgroup_bytes": memory,
                            "swap_bytes": swap,
                            "gpu_mib": gpu,
                        }
                    )
                    + "\n"
                )
                log.flush()
                reason = "termination requested" if interrupted else stop_reason(available, disk, gpu, elapsed, timeout)
                if reason:
                    break
                time.sleep(0.5)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        reason = f"guard monitor failed: {type(error).__name__}"
        raise
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        peak_memory = max(peak_memory, int((cgroup / "memory.peak").read_text()))
        result = {
            "returncode": proc.returncode,
            "stop_reason": reason,
            "elapsed_seconds": time.monotonic() - start,
            "minimum_available_bytes": minimum_ram,
            "minimum_disk_free_bytes": minimum_disk,
            "peak_cgroup_bytes": peak_memory,
            "peak_swap_bytes": peak_swap,
            "peak_gpu_mib": peak_gpu,
        }
        (artifact / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Guard result: {artifact / 'result.json'}", flush=True)
    return 0 if proc.returncode == 0 and reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
