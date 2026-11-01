"""Fixed-work SubprocessRuntime fork_server comparison using a local mock model."""

import asyncio
import hashlib
import json
import statistics
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from types import SimpleNamespace

from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM_SOURCE, launch_chat_program
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime

N = 8
CONCURRENCY = 2
BLOCKS = (False, True, True, False, True, False, False, True, True, False, True, False)
MODEL = "mock-fixed-work"
NAMES = (
    "Ada Lovelace", "Bjarne Stroustrup", "Carl Friedrich Gauss", "Donald Knuth",
    "Edsger Dijkstra", "Frances Allen", "Grace Hopper", "Hedy Lamarr",
)
REQUESTS: list[dict] = []
REQUEST_LOCK = Lock()


def sorted_answer(user: str) -> str:
    names = user.splitlines()[1:]
    ordered = sorted(names, key=lambda name: name.rsplit(" ", 1)[-1])
    return "<alphabetical_sorted>\n" + "\n".join(ordered) + "\n</alphabetical_sorted>"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        received = time.perf_counter()
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")
        answer = sorted_answer(user)
        def event(delta: dict, finish: str | None = None, **extra) -> bytes:
            payload = {"id": "mock-completion", "object": "chat.completion.chunk",
                       "created": 0, "model": body["model"],
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                       **extra}
            return ("data: " + json.dumps(payload) + "\n\n").encode()

        chunks = (
            event({"role": "assistant"}),
            event({"content": answer}),
            event({}, "stop"),
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for data in chunks:
            self.wfile.write(data)
        self.wfile.write(("data: " + json.dumps({
            "id": "mock-completion", "object": "chat.completion.chunk", "created": 0,
            "model": body["model"], "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        with REQUEST_LOCK:
            REQUESTS.append({"body": body, "answer": answer,
                             "first": received, "done": time.perf_counter()})


def start_mock() -> tuple[ThreadingHTTPServer, Thread, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}/v1"


def prompt(i: int) -> tuple[str, str]:
    rotated = NAMES[i % len(NAMES):] + NAMES[:i % len(NAMES)]
    system = "Sort the supplied names alphabetically by last name. Return only the tagged list."
    user = "Sort these authors alphabetically by last name, one per line:\n" + "\n".join(rotated)
    return system, user


async def one(i: int, fork: bool, endpoint: str) -> dict:
    runtime = SubprocessRuntime(SubprocessConfig(fork_server=fork))
    await runtime.start()
    try:
        system, user = prompt(i)
        started = time.perf_counter()
        result = await launch_chat_program(
            CHAT_PROGRAM_SOURCE, HarnessConfig(), SimpleNamespace(model=MODEL), None,
            runtime, endpoint, "mock-secret", {}, system, user,
        )
        ended = time.perf_counter()
        return {"i": i, "started": started, "ended": ended,
                "exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr}
    finally:
        await runtime.stop()


async def block(fork: bool, endpoint: str) -> dict:
    with REQUEST_LOCK:
        REQUESTS.clear()
    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(i: int) -> dict:
        async with sem:
            return await one(i, fork, endpoint)

    started = time.perf_counter()
    rows = await asyncio.gather(*(bounded(i) for i in range(N)))
    wall = time.perf_counter() - started
    with REQUEST_LOCK:
        requests = list(REQUESTS)
    if len(requests) != N:
        raise RuntimeError(f"expected {N} mock requests, received {len(requests)}")
    requests_by_prompt = {r["body"]["messages"][-1]["content"]: r for r in requests}
    expected_users = {prompt(i)[1] for i in range(N)}
    if len(expected_users) != N or set(requests_by_prompt) != expected_users:
        raise RuntimeError("mock request prompt set did not match unique fixed-work prompts")
    if len(requests_by_prompt) != len(requests):
        raise RuntimeError("duplicate mock request prompt detected")
    for row in rows:
        _, user = prompt(row["i"])
        request = requests_by_prompt[user]
        row["launch_to_first_request_s"] = request["first"] - row["started"]
        row["episode_s"] = row["ended"] - row["started"]
        row["after_response_s"] = row["ended"] - request["done"]
    return {"fork_server": fork, "wall_s": wall, "rows": rows,
            "requests": sorted((r["body"] for r in requests),
                               key=lambda body: body["messages"][-1]["content"]),
            "mock_answers": sorted((r["answer"] for r in requests))}


def stats(values: list[float]) -> dict:
    return {"median_s": statistics.median(values), "mean_s": statistics.mean(values)}


async def main() -> None:
    server, thread, endpoint = start_mock()
    try:
        # First invocation of each arm is recorded as setup and excluded from timed blocks.
        setup = {}
        for label, fork in (("off", False), ("on", True)):
            t0 = time.perf_counter()
            result = await one(0, fork, endpoint)
            setup[label] = {"first_invocation_s": time.perf_counter() - t0,
                            "exit_code": result["exit_code"], "stderr": result["stderr"]}
            if result["exit_code"] != 0:
                raise RuntimeError(f"{label} setup invocation failed: {result}")
        # Confirm warm fork server exists before interleaved measurements begin.
        if not SubprocessRuntime._forks:
            raise RuntimeError("fork-server setup did not leave a warm server")
        measurements = []
        for index, fork in enumerate(BLOCKS):
            measurements.append({"block": index, **(await block(fork, endpoint))})

        reference = measurements[0]
        mismatches = []
        for current in measurements[1:]:
            if current["requests"] != reference["requests"]:
                mismatches.append({"block": current["block"], "kind": "request_bodies"})
            if current["mock_answers"] != reference["mock_answers"]:
                mismatches.append({"block": current["block"], "kind": "mock_answers"})
            ref_rows = {r["i"]: r for r in reference["rows"]}
            for row in current["rows"]:
                ref = ref_rows[row["i"]]
                if (row["exit_code"], row["stdout"], row["stderr"]) != (
                    ref["exit_code"], ref["stdout"], ref["stderr"]
                ):
                    mismatches.append({"block": current["block"], "prompt": row["i"], "kind": "episode"})

        summary = {}
        for label, fork in (("off", False), ("on", True)):
            arm = [m for m in measurements if m["fork_server"] is fork]
            rows = [row for m in arm for row in m["rows"]]
            summary[label] = {
                "blocks": len(arm), "wall_s_per_block": stats([m["wall_s"] for m in arm]),
                **{key: stats([r[key] for r in rows]) for key in
                   ("launch_to_first_request_s", "episode_s", "after_response_s")},
                "requests": sum(len(m["requests"]) for m in arm),
                "episodes": len(rows),
                "successful_episodes": sum(r["exit_code"] == 0 for r in rows),
                "nonzero_exit_count": sum(r["exit_code"] != 0 for r in rows),
            }
            if summary[label]["nonzero_exit_count"]:
                raise RuntimeError(f"{label} arm has failed episodes")
        report = {
            "scope": "runtime microbenchmark with local deterministic mock SSE; no live model/GPU/optimizer",
            "fixed_work": {"episodes_per_block": N, "concurrency": CONCURRENCY,
                           "names_per_prompt": len(NAMES), "block_order": ["on" if x else "off" for x in BLOCKS]},
            "setup_excluded_from_timing": setup,
            "summary": summary, "equivalence_mismatches": mismatches,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "blocks": measurements,
        }
        if mismatches:
            raise RuntimeError(f"equivalence mismatches: {mismatches}")
        report_path = Path(__file__).with_name("results.json")
        if report_path.exists():
            raise FileExistsError(f"refusing to overwrite {report_path}")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        SubprocessRuntime.stop_fork_servers()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
