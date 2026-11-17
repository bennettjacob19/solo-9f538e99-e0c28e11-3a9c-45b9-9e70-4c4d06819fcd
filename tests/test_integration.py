"""End-to-end fault-injection tests against real uvicorn processes.

Spins up one coordinator + two independent worker processes (each with its own
temporary data directory, exactly like the Docker topology) and verifies:

1. clean distributed training equals the single-process reference trajectory
   bit-for-bit;
2. killing a worker mid-training pauses the group, and resume reproduces the
   exact uninterrupted trajectory (no skipped / double-applied round);
3. redoing an unconfirmed round after coordinator restart stays exact;
4. stale-generation messages are fenced (409) and cannot advance a new group;
5. downloaded weights only ever reflect confirmed rounds;
6. concurrent uploads admit exactly one task: the loser gets an explicit 409
   and its samples never reach any worker;
7. workers fence shard loads with a compare-and-swap token, so a late/foreign
   shard push can never overwrite the accepted task;
8. a task accepted while one side is unreachable recovers via the ordinary
   start/resume path (shard re-delivery), with no CSV re-upload.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from app import lrmath
from app.storage import content_hash

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
ROUNDS = 30


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Service:
    def __init__(self, name, module, port, env, health_path="/health"):
        self.name, self.module, self.port, self.env = name, module, port, env
        self.health_path = health_path
        self.proc: subprocess.Popen | None = None

    def start(self):
        import site
        env = dict(os.environ)
        env.update(self.env)
        # Use -P (cwd excluded from sys.path, guarding against planted modules)
        # rather than -I, which would also drop PYTHONPATH and user site
        # packages. The app dir and locally installed deps are supplied here.
        extra = [str(APP), site.getusersitepackages()]
        env["PYTHONPATH"] = os.pathsep.join(extra) + os.pathsep + env.get(
            "PYTHONPATH", "")
        self.proc = subprocess.Popen(
            [sys.executable, "-P", "-m", "uvicorn",
             f"{self.module}:app", "--host", "127.0.0.1",
             "--port", str(self.port)],
            cwd=APP, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self._wait_up()

    def _wait_up(self, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                out = self.proc.stdout.read() if self.proc.stdout else ""
                raise RuntimeError(f"{self.name} exited early:\n{out}")
            try:
                r = httpx.get(
                    f"http://127.0.0.1:{self.port}{self.health_path}",
                    timeout=1.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.15)
        raise TimeoutError(f"{self.name} did not come up")

    def hard_kill(self):
        assert self.proc
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.fixture()
def cluster(tmp_path):
    p1, p2, pc = _free_port(), _free_port(), _free_port()
    d1, d2, dc = (tmp_path / "w1"), (tmp_path / "w2"), (tmp_path / "coord")
    for d in (d1, d2, dc):
        d.mkdir()
    w1 = Service("worker1", "worker", p1,
                 {"WORKER_ID": "worker1", "WORKER_DATA": str(d1)})
    w2 = Service("worker2", "worker", p2,
                 {"WORKER_ID": "worker2", "WORKER_DATA": str(d2)})
    coord = Service("coordinator", "coordinator", pc, {
        "COORD_DATA": str(dc),
        "WORKER1_URL": f"http://127.0.0.1:{p1}",
        "WORKER2_URL": f"http://127.0.0.1:{p2}",
    }, health_path="/api/status")
    w1.start(); w2.start(); coord.start()
    yield {
        "coord": f"http://127.0.0.1:{pc}",
        "coord_svc": coord, "w1": w1, "w2": w2,
        "paths": {"w1": d1, "w2": d2, "coord": dc},
    }
    for s in (w1, w2, coord):
        s.stop()


def _dataset_csv(tmp_path: Path, n: int = 40) -> Path:
    p = tmp_path / "data.csv"
    p.write_text(_csv_text(n, seed=7), encoding="utf-8")
    return p


def _csv_text(n: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    z = 1.2 * x[:, 0] - 0.6 * x[:, 1]
    y = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
    lines = ["x1,x2,label"]
    for (a, b), c in zip(x, y):
        lines.append(f"{a:.10f},{b:.10f},{c}")
    return "\n".join(lines)


def _parse_csv_text(text: str):
    lines = [l for l in text.splitlines() if l.strip()]
    rows = lines[1:]  # drop the header row
    x = np.array([[float(v) for v in l.split(",")[:2]] for l in rows])
    y = np.array([float(l.split(",")[2]) for l in rows])
    return x, y


def _wait_status(base, predicate, timeout=40):
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = httpx.get(base + "/api/status", timeout=5).json()
        if predicate(last):
            return last
        time.sleep(0.25)
    raise AssertionError(f"condition not met; last={json.dumps(last)[:600]}")


def _create_and_start(base, csv_path, rounds=ROUNDS):
    with open(csv_path, "rb") as f:
        r = httpx.post(base + f"/api/job?rounds={rounds}",
                       files={"file": ("data.csv", f, "text/csv")},
                       timeout=15)
    r.raise_for_status()
    httpx.post(base + "/api/start", timeout=10).raise_for_status()


# --------------------------------------------------------------------- tests
def test_clean_run_matches_reference(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    st = _wait_status(base, lambda d: d["status"] == "completed")
    assert st["confirmed_round"] == ROUNDS

    ref = httpx.get(base + "/api/reference", timeout=10).json()
    # Exact, bit-for-bit equality with uninterrupted single-process training.
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert np.allclose(st["w"], ref["weights"][-1], atol=0.0, rtol=0.0)
    assert np.allclose(st["v"], ref["momentum"][-1], atol=0.0, rtol=0.0)

    # Weights download corresponds to the confirmed round only.
    wdoc = httpx.get(base + "/api/weights", timeout=5).json()
    assert wdoc["confirmed_round"] == ROUNDS
    assert wdoc["weights"]["w1"] == st["w"][1]


def test_worker_kill_and_resume_is_numerically_exact(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    # Let some rounds confirm, then SIGKILL worker1 mid-flight.
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 8)
    confirmed_at_kill = st["confirmed_round"]
    cluster["w1"].hard_kill()

    st = _wait_status(base, lambda d: d["status"] == "paused")
    assert st["confirmed_round"] == confirmed_at_kill  # nothing half-counted

    # The killed worker's durable state never exceeds the confirmed round by
    # more than one (a commit may have landed just before the kill).
    state1 = json.loads((cluster["paths"]["w1"] / "state.json").read_text())
    assert state1["round"] <= confirmed_at_kill + 1

    # Restart the worker process (fresh generation-neutral process).
    cluster["w1"].start()
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed")

    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k], f"round {k} differs"
    np.testing.assert_array_equal(np.asarray(st["w"]),
                                  np.asarray(ref["weights"][-1]))
    # Exactly one loss entry per round: no duplicate application.
    rounds_seen = [p["round"] for p in st["loss_history"]]
    assert rounds_seen == list(range(ROUNDS + 1))


def test_coordinator_restart_mid_round_redoes_without_skip(cluster, tmp_path):
    base = cluster["coord"]
    # Add delay so a round is almost certainly in flight at kill time.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.4}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=12)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 4)
    confirmed = st["confirmed_round"]
    cluster["coord_svc"].hard_kill()
    time.sleep(0.8)  # let at least one delayed in-flight round drift
    cluster["coord_svc"].start()

    st = httpx.get(base + "/api/status", timeout=5).json()
    assert st["status"] == "paused"
    assert st["confirmed_round"] == confirmed
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=60)

    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert [p["round"] for p in st["loss_history"]] == list(range(13))


def test_stale_generation_messages_are_fenced(cluster, tmp_path):
    base = cluster["coord"]
    # Slow each round down so the pause reliably lands while training runs.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.5}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=10)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 2)
    old_gen = st["gen"]
    task_id = st["task_id"]

    # Pause and resume -> a strictly newer generation is installed.
    deadline = time.time() + 15
    while True:
        r = httpx.post(base + "/api/pause", timeout=5)
        if r.status_code == 200:
            break
        if httpx.get(base + "/api/status").json()["status"] == "completed":
            raise AssertionError("training completed before pause took effect")
        assert time.time() < deadline
        time.sleep(0.3)
    _wait_status(base, lambda d: d["status"] == "paused")
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st2 = _wait_status(base, lambda d: d["gen"] > old_gen)
    new_gen = st2["gen"]

    # A late message from the old training group must be rejected (409) and
    # must not mutate any state. Reach the worker directly on its own port.
    w1_port = cluster["w1"].port
    r = httpx.post(f"http://127.0.0.1:{w1_port}/gradient",
                   json={"task_id": task_id, "gen": old_gen,
                         "round": st2["confirmed_round"] + 1},
                   timeout=5)
    assert r.status_code == 409

    # Unknown task id is also rejected.
    r2 = httpx.post(f"http://127.0.0.1:{w1_port}/commit",
                    json={"task_id": "deadbeef", "gen": new_gen,
                          "round": 1, "grad": [0, 0, 0]}, timeout=5)
    assert r2.status_code in (409, 410)

    final = _wait_status(base, lambda d: d["status"] == "completed")
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(final["loss_history"]):
        assert point["loss"] == ref["loss"][k]


def test_full_cluster_reboot_keeps_durable_state(cluster, tmp_path):
    base = cluster["coord"]
    # Slow rounds so the kill reliably lands mid-training, not after completion.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.4}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=15)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 4)
    confirmed = st["confirmed_round"]
    # Kill every process, then bring the whole cluster back up.
    cluster["coord_svc"].hard_kill()
    cluster["w1"].hard_kill()
    cluster["w2"].hard_kill()
    for s in (cluster["w1"], cluster["w2"], cluster["coord_svc"]):
        s.start()
    st = httpx.get(base + "/api/status", timeout=10).json()
    # More rounds may have confirmed between observation and the kill; the
    # durable invariant that matters is that progress is never lost and the
    # reboot resumes from a confirmed round >= the one we observed.
    assert st["confirmed_round"] >= confirmed
    assert st["status"] in ("paused", "idle", "completed")
    if st["status"] != "completed":
        httpx.post(base + "/api/resume", timeout=15).raise_for_status()
        st = _wait_status(base, lambda d: d["status"] == "completed",
                          timeout=60)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert [p["round"] for p in st["loss_history"]] == list(range(16))


def test_csv_validation_rules(cluster, tmp_path):
    base = cluster["coord"]
    good = _dataset_csv(tmp_path, n=20)
    # Below the 2-row minimum: header + a single data row -> 400.
    p = tmp_path / "small.csv"
    p.write_text("x1,x2,label\n1,2,0\n")
    with open(p, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=5",
                       files={"file": ("small.csv", f, "text/csv")})
    assert r.status_code == 400
    # The minimum allowed size (2 rows, both classes) is accepted.
    pmin = tmp_path / "min.csv"
    pmin.write_text("1,2,0\n3,4,1\n")  # no header, exactly 2 rows
    with open(pmin, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=3",
                       files={"file": ("min.csv", f, "text/csv")})
    assert r.status_code in (200, 202)
    httpx.post(base + "/api/reset", timeout=5)
    # Bad label.
    p2 = tmp_path / "badlabel.csv"
    lines = good.read_text().splitlines()
    head, body = lines[0], lines[1:]
    body[0] = body[0][:-1] + "9"
    p2.write_text("\n".join([head] + body))
    with open(p2, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=5",
                       files={"file": ("b.csv", f, "text/csv")})
    assert r.status_code == 400
    # Rounds out of range.
    with open(good, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=101",
                       files={"file": ("g.csv", f, "text/csv")})
    assert r.status_code == 400
    # Over the 200-row maximum.
    big = tmp_path / "big.csv"
    rows = ["x1,x2,label"] + [f"{i},1,{i % 2}" for i in range(201)]
    big.write_text("\n".join(rows))
    with open(big, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=5",
                       files={"file": ("big.csv", f, "text/csv")})
    assert r.status_code == 400


def test_download_weights_only_confirmed_and_save_keeps_last(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=12)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 5)
    wdoc = httpx.get(base + "/api/weights", timeout=5).json()
    assert wdoc["confirmed_round"] == st["confirmed_round"]
    # On-disk immutable artifact plus latest pointer exist.
    cdir = cluster["paths"]["coord"] / "weights"
    assert (cdir / f"weights_round_{st['confirmed_round']:04d}.json").exists()
    assert (cdir / "weights_latest.json").exists()
    final = _wait_status(base, lambda d: d["status"] == "completed")
    wdoc = httpx.get(base + "/api/weights", timeout=5).json()
    assert wdoc["confirmed_round"] == final["confirmed_round"] == 12


def test_worker_commit_dedup_and_prepare_rollback(cluster, tmp_path):
    """Direct worker protocol: a retried commit applies once, and /prepare
    rolls an unconfirmed advance back to the last confirmed round."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=8)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 3)
    task_id, gen = st["task_id"], st["gen"]
    w1_port = cluster["w1"].port
    url = f"http://127.0.0.1:{w1_port}"

    h = httpx.get(url + "/health", timeout=5).json()
    at = h["round"]
    # Gradient for the next round.
    g = httpx.post(url + "/gradient",
                   json={"task_id": task_id, "gen": gen, "round": at + 1},
                   timeout=5).json()
    grad = [0.01, -0.02, 0.03]
    c1 = httpx.post(url + "/commit",
                    json={"task_id": task_id, "gen": gen,
                          "round": at + 1, "grad": grad}, timeout=5).json()
    assert c1["round"] == at + 1 and c1["dedup"] is False
    w_after, v_after = c1["w"], c1["v"]
    # Retried commit for the same round must NOT apply the step a second time.
    c2 = httpx.post(url + "/commit",
                    json={"task_id": task_id, "gen": gen,
                          "round": at + 1, "grad": grad}, timeout=5).json()
    assert c2["dedup"] is True and c2["w"] == w_after and c2["v"] == v_after

    # A newer generation's /prepare rolls the worker back to the confirmed
    # round (discarding the unconfirmed local advance above).
    new_gen = gen + 1
    p = httpx.post(url + "/prepare",
                   json={"task_id": task_id, "gen": new_gen,
                         "round": at, "w": st["w"], "v": st["v"]},
                   timeout=5).json()
    assert p["gen"] == new_gen and p["round"] == at
    h2 = httpx.get(url + "/health", timeout=5).json()
    assert h2["round"] == at and h2["gen"] == new_gen

    # Old-generation traffic is now fenced.
    r = httpx.post(url + "/gradient",
                   json={"task_id": task_id, "gen": gen, "round": at + 1},
                   timeout=5)
    assert r.status_code == 409
    # No skipping rounds: committing round at+2 before at+1 is rejected.
    r2 = httpx.post(url + "/commit",
                    json={"task_id": task_id, "gen": new_gen,
                          "round": at + 2, "grad": grad}, timeout=5)
    assert r2.status_code == 409


def test_delay_endpoint_survives_and_rounds_still_exact(cluster, tmp_path):
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.2}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=6)
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=60)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]


# ------------------------------------------------------- concurrency control
def test_concurrent_uploads_single_winner(cluster, tmp_path):
    """Two engineers uploading different CSVs at the same time: exactly one
    task is admitted, the other gets an explicit 409, responses never carry a
    foreign task id, and both workers end up holding the WINNER's shards."""
    base = cluster["coord"]
    csv_a = _csv_text(40, seed=7)
    csv_b = _csv_text(35, seed=21)  # different content AND different row count

    async def race():
        async with httpx.AsyncClient(base_url=base, timeout=30) as c:
            return await asyncio.gather(
                c.post("/api/job?rounds=6",
                       files={"file": ("a.csv", csv_a, "text/csv")}),
                c.post("/api/job?rounds=6",
                       files={"file": ("b.csv", csv_b, "text/csv")}))

    r_a, r_b = asyncio.run(race())
    codes = sorted([r_a.status_code, r_b.status_code])
    assert codes == [200, 409], f"expected one winner one conflict, got {codes}"
    loser = r_a if r_a.status_code == 409 else r_b
    assert "冲突" in loser.json()["detail"] or "拒绝" in loser.json()["detail"]

    win_resp, win_csv = ((r_a, csv_a) if r_a.status_code == 200
                         else (r_b, csv_b))
    task_id = win_resp.json()["task_id"]

    # Coordinator state and BOTH workers must belong to the winner's task.
    st = httpx.get(base + "/api/status", timeout=5).json()
    assert st["task_id"] == task_id and st["status"] == "idle"
    x_win, y_win = _parse_csv_text(win_csv)
    n_win = len(y_win)
    n_seen = 0
    for wid in ("worker1", "worker2"):
        h = st["workers"][wid]
        assert h["alive"] and h["task_id"] == task_id
        assert h["shard_match"] is True
        n_seen += h["n"]
    assert n_seen == n_win  # full coverage, no foreign samples mixed in

    # The admitted task trains to completion and matches the winner's CSV
    # bit-for-bit (any leaked sample from the loser would diverge here).
    httpx.post(base + "/api/start", timeout=15).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed")
    ws, vs, losses = lrmath.reference_train(x_win, y_win, 6)
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == losses[k], f"round {k} differs"
    np.testing.assert_array_equal(np.asarray(st["w"]), np.asarray(ws[-1]))


def test_concurrent_upload_storm_admits_exactly_one(cluster, tmp_path):
    """Stress variant: 8 simultaneous uploads -> 1 accepted, 7 conflicts."""
    base = cluster["coord"]
    csvs = {f"u{k}.csv": _csv_text(30 + k, seed=100 + k) for k in range(8)}

    async def storm():
        async with httpx.AsyncClient(base_url=base, timeout=30) as c:
            return await asyncio.gather(*(
                c.post("/api/job?rounds=4",
                       files={"file": (name, text, "text/csv")})
                for name, text in csvs.items()))

    rs = asyncio.run(storm())
    codes = [r.status_code for r in rs]
    assert codes.count(200) == 1, codes
    assert codes.count(409) == len(csvs) - 1, codes
    winner_task = next(r.json()["task_id"] for r in rs if r.status_code == 200)
    st = httpx.get(base + "/api/status", timeout=5).json()
    assert st["task_id"] == winner_task
    for wid in ("worker1", "worker2"):
        assert st["workers"][wid]["task_id"] == winner_task
        assert st["workers"][wid]["shard_match"] is True


def test_worker_load_cas_fencing(cluster, tmp_path):
    """Worker-level fencing: a late/foreign shard push must not overwrite the
    accepted task; authorised replacement requires naming the held task."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=4)
    st = _wait_status(base, lambda d: d["status"] == "completed")
    task_id = st["task_id"]
    w1 = f"http://127.0.0.1:{cluster['w1'].port}"

    shard = json.loads((cluster["paths"]["w1"] / "shard.json").read_text())
    rows, shard_hash = shard["rows"], shard["shard_hash"]

    # 1) Idempotent re-delivery of the SAME shard: accepted, state preserved.
    r = httpx.post(w1 + "/load", json={
        "task_id": task_id, "worker_id": "worker1", "rows": rows,
        "shard_hash": shard_hash, "expected_current": None}, timeout=5)
    assert r.status_code == 200 and r.json()["reloaded"] is True
    assert r.json()["round"] == 4  # training state NOT wiped

    # 2) Foreign task without a CAS token -> 409, nothing changes.
    foreign_rows = [{"x": [1.0, 2.0], "y": 1.0}, {"x": [3.0, 4.0], "y": 0.0}]
    foreign_hash = content_hash(foreign_rows)
    for bad_token in (None, "some-other-task"):
        r = httpx.post(w1 + "/load", json={
            "task_id": "foreign-task", "worker_id": "worker1",
            "rows": foreign_rows, "shard_hash": foreign_hash,
            "expected_current": bad_token}, timeout=5)
        assert r.status_code == 409
    h = httpx.get(w1 + "/health", timeout=5).json()
    assert h["task_id"] == task_id and h["round"] == 4

    # 3) Same task id but different content -> 409 (corruption is loud).
    r = httpx.post(w1 + "/load", json={
        "task_id": task_id, "worker_id": "worker1", "rows": foreign_rows,
        "shard_hash": foreign_hash, "expected_current": task_id}, timeout=5)
    assert r.status_code == 409

    # 4) Content hash not matching the actual rows -> 400.
    r = httpx.post(w1 + "/load", json={
        "task_id": "t2", "worker_id": "worker1", "rows": foreign_rows,
        "shard_hash": "0" * 64, "expected_current": task_id}, timeout=5)
    assert r.status_code == 400

    # 5) Authorised replacement: CAS token names the held task -> accepted.
    r = httpx.post(w1 + "/load", json={
        "task_id": "foreign-task", "worker_id": "worker1",
        "rows": foreign_rows, "shard_hash": foreign_hash,
        "expected_current": task_id}, timeout=5)
    assert r.status_code == 200
    h = httpx.get(w1 + "/health", timeout=5).json()
    assert h["task_id"] == "foreign-task" and h["round"] == 0
    # Durable files reflect the replacement (crash-safe).
    disk = json.loads((cluster["paths"]["w1"] / "shard.json").read_text())
    assert disk["task_id"] == "foreign-task"


def test_upload_with_worker_down_recovers_without_reupload(cluster, tmp_path):
    """One side unreachable at upload time: the task is still admitted (202),
    a concurrent second upload is rejected, and the ordinary start/resume
    path re-delivers the missing shard - no CSV re-upload, no foreign rows."""
    base = cluster["coord"]
    cluster["w2"].hard_kill()

    csv_path = _dataset_csv(tmp_path, n=30)
    with open(csv_path, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=10",
                       files={"file": ("d.csv", f, "text/csv")}, timeout=20)
    assert r.status_code == 202  # admitted, shard delivery pending
    st = r.json()
    assert st["status"] == "paused" and st["last_error"]
    task_id = st["task_id"]

    # A second upload while one side is down must be rejected, not queued.
    other = _csv_text(28, seed=99)
    r2 = httpx.post(base + "/api/job?rounds=5",
                    files={"file": ("o.csv", other, "text/csv")}, timeout=10)
    assert r2.status_code == 409

    # Starting while worker2 is still down fails cleanly and keeps the task.
    r3 = httpx.post(base + "/api/start", timeout=20)
    assert r3.status_code == 503
    st = httpx.get(base + "/api/status", timeout=5).json()
    assert st["status"] == "paused" and st["task_id"] == task_id

    # Bring worker2 back: the ordinary recovery path must heal the shard
    # from the durable job and train the ORIGINAL task to completion.
    cluster["w2"].start()
    httpx.post(base + "/api/start", timeout=20).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed")
    assert st["task_id"] == task_id
    assert st["confirmed_round"] == 10

    h2 = httpx.get(f"http://127.0.0.1:{cluster['w2'].port}/health",
                   timeout=5).json()
    assert h2["task_id"] == task_id and h2["n"] == 30 - (30 + 1) // 2
    # Exactly the uploaded data was trained (reference is bit-identical).
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    x, y = _parse_csv_text(csv_path.read_text())
    ws, _, _ = lrmath.reference_train(x, y, 10)
    np.testing.assert_array_equal(np.asarray(st["w"]), np.asarray(ws[-1]))


def test_reset_clears_workers_for_next_upload(cluster, tmp_path):
    """Reset clears both workers; the next upload starts from empty processes
    even though they still hold the previous task's shard."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=4)
    _wait_status(base, lambda d: d["status"] == "completed")

    httpx.post(base + "/api/reset", timeout=10).raise_for_status()
    st = httpx.get(base + "/api/status", timeout=5).json()
    assert "task_id" not in st or st.get("status") is None
    for wid in ("worker1", "worker2"):
        assert st["workers"][wid]["task_id"] is None

    # A fresh upload afterwards is admitted and consistent everywhere.
    csv_path = _dataset_csv(tmp_path, n=22)
    _create_and_start(base, csv_path, rounds=5)
    st = _wait_status(base, lambda d: d["status"] == "completed")
    assert st["confirmed_round"] == 5
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]


def test_reupload_after_completion_replaces_shards(cluster, tmp_path):
    """Uploading a new CSV after the previous task COMPLETED is admitted and
    replaces the workers' shards via the authorised CAS path (no reset)."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path, n=30), rounds=4)
    st = _wait_status(base, lambda d: d["status"] == "completed")
    old_task = st["task_id"]

    # New upload without an explicit reset: admitted because the previous
    # task is completed.
    csv2 = tmp_path / "second.csv"
    csv2.write_text(_csv_text(26, seed=42), encoding="utf-8")
    with open(csv2, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=5",
                       files={"file": ("s.csv", f, "text/csv")}, timeout=15)
    assert r.status_code == 200
    new_task = r.json()["task_id"]
    assert new_task != old_task

    st = httpx.get(base + "/api/status", timeout=5).json()
    for wid in ("worker1", "worker2"):
        h = st["workers"][wid]
        assert h["task_id"] == new_task and h["shard_match"] is True
        assert h["round"] == 0  # replaced shards start from round zero

    httpx.post(base + "/api/start", timeout=15).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed")
    x, y = _parse_csv_text(csv2.read_text())
    ws, _, losses = lrmath.reference_train(x, y, 5)
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == losses[k]
    np.testing.assert_array_equal(np.asarray(st["w"]), np.asarray(ws[-1]))
