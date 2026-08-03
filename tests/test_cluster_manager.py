"""ClusterModelServerManager hardening (family cluster-job standard §5):
upfront config validation (T1), durable job ledger + orphan reaping (T2),
atexit/signal teardown (T3), eviction hysteresis + recovery (T4), per-model
SLURM timeouts (T5), Condition-based pool, sbatch retry. NO network, NO
SLURM: subprocess and health checks are monkeypatched/faked."""

import json
import signal
import socket
import threading

import pytest

import llm_utils.cluster_server_manager as csm
from llm_utils.cluster_server_manager import ClusterModelServerManager
from llm_utils.llm_model import LLMModel

MODEL = LLMModel.LLAMA3_1_8B_CLUSTER
MODEL_B = LLMModel.QWEN2_5_7B_INSTRUCT


def make_config(**over):
    cfg = {
        "num_instances": 1,
        "port": 8000,
        "partition": "gpu",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 4096,
        "num_gpus": 1,
        "cpus_per_task": 8,
        "mem_gb": 64,
        "time_limit": "04:00:00",
        "hf_home": "/scratch/hf",
        "env_setup": ["module load cuda", "source .venv/bin/activate"],
        "install_signal_handlers": False,
    }
    cfg.update(over)
    return cfg


@pytest.fixture
def manager(monkeypatch, tmp_path):
    """A manager whose monitor thread and teardown registration are inert,
    running in an isolated tmp CWD (ledgers land in ./logs)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ClusterModelServerManager, "_monitor_loop", lambda self: None)
    monkeypatch.setattr(
        ClusterModelServerManager, "_register_teardown",
        lambda self, config: None)
    m = ClusterModelServerManager()
    yield m
    # Nothing must survive to the real atexit hook.
    m._jobs.clear()


def submit_recorder(monkeypatch, manager):
    """Monkeypatch _submit_sbatch to record calls and mint job ids."""
    submitted = []

    def fake_submit(self, sbatch_path):
        submitted.append(sbatch_path)
        return str(1000 + len(submitted))

    monkeypatch.setattr(ClusterModelServerManager, "_submit_sbatch", fake_submit)
    return submitted


# ==================== T1: upfront validation ====================

class TestUpfrontValidation:
    def test_missing_keys_all_named_and_nothing_submitted(self, manager, monkeypatch):
        submitted = submit_recorder(monkeypatch, manager)
        cfg = make_config()
        del cfg["partition"], cfg["mem_gb"]
        with pytest.raises(ValueError) as exc:
            manager.start_server(MODEL, cfg)
        assert "partition" in str(exc.value)
        assert "mem_gb" in str(exc.value)
        assert submitted == []
        assert MODEL not in manager._jobs
        assert MODEL not in manager.model_configs

    def test_conda_keys_required_only_without_env_setup(self, manager, monkeypatch):
        submit_recorder(monkeypatch, manager)
        cfg = make_config()
        del cfg["env_setup"]
        with pytest.raises(ValueError) as exc:
            manager.start_server(MODEL, cfg)
        assert "conda_env" in str(exc.value)
        assert "cuda_module" in str(exc.value)

    def test_bad_num_instances_rejected(self, manager, monkeypatch):
        submitted = submit_recorder(monkeypatch, manager)
        with pytest.raises(ValueError, match="num_instances"):
            manager.start_server(MODEL, make_config(num_instances=0))
        assert submitted == []

    def test_all_scripts_generated_before_first_submit(self, manager, monkeypatch):
        events = []
        monkeypatch.setattr(
            ClusterModelServerManager, "_generate_sbatch",
            lambda self, model, config, instance_id=0:
                events.append("gen") or f"script_{instance_id}")
        monkeypatch.setattr(
            ClusterModelServerManager, "_submit_sbatch",
            lambda self, path: events.append("sub") or str(len(events)))
        manager.start_server(MODEL, make_config(num_instances=3))
        assert events == ["gen", "gen", "gen", "sub", "sub", "sub"]


# ==================== T2: durable job ledger ====================

class TestJobLedger:
    def test_ledger_written_at_submission(self, manager, monkeypatch, tmp_path):
        submit_recorder(monkeypatch, manager)
        manager.start_server(MODEL, make_config(num_instances=2))

        ledgers = list((tmp_path / "logs").glob("slurm_jobs_*.json"))
        assert len(ledgers) == 1
        data = json.loads(ledgers[0].read_text())
        assert data["closed"] is False
        assert data["host"] == socket.gethostname()
        assert [j["job_id"] for j in data["jobs"]] == ["1001", "1002"]
        assert all(j["cancelled"] is False for j in data["jobs"])

    def test_shutdown_closes_ledger_and_cancels(self, manager, monkeypatch):
        submit_recorder(monkeypatch, manager)
        cancelled = []
        monkeypatch.setattr(
            ClusterModelServerManager, "_scancel",
            lambda self, job_id: cancelled.append(job_id))
        manager.start_server(MODEL, make_config(num_instances=2))
        manager.shutdown_all()

        assert cancelled == ["1001", "1002"]
        data = json.loads(manager._ledger_path().read_text())
        assert data["closed"] is True
        assert all(j["cancelled"] for j in data["jobs"])

    def test_shutdown_model_marks_its_jobs_cancelled(self, manager, monkeypatch):
        submit_recorder(monkeypatch, manager)
        monkeypatch.setattr(
            ClusterModelServerManager, "_scancel", lambda self, job_id: None)
        manager.start_server(MODEL, make_config(num_instances=1))
        manager.shutdown_model(MODEL)
        data = json.loads(manager._ledger_path().read_text())
        assert data["closed"] is False  # run still alive, only one model down
        assert data["jobs"][0]["cancelled"] is True


def write_stale_ledger(tmp_path, host, pid, job_ids, closed=False):
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / "slurm_jobs_20260101_000000_99999.json"
    path.write_text(json.dumps({
        "run_id": "20260101_000000_99999",
        "host": host,
        "pid": pid,
        "closed": closed,
        "jobs": [
            {"job_id": j, "model": "OLD", "cancelled": False} for j in job_ids
        ],
    }))
    return path


class TestOrphanReaping:
    def test_dead_pid_same_host_reaped_and_closed(self, manager, monkeypatch, tmp_path):
        stale = write_stale_ledger(tmp_path, socket.gethostname(), 99999, ["555", "556"])
        submit_recorder(monkeypatch, manager)
        monkeypatch.setattr(csm, "_pid_alive", lambda pid: False)
        # 555 still queued -> scancel; 556 already gone -> skip.
        monkeypatch.setattr(
            ClusterModelServerManager, "_get_job_state",
            lambda self, job_id, config: "RUNNING" if job_id == "555" else None)
        reaped = []
        monkeypatch.setattr(
            ClusterModelServerManager, "_scancel",
            lambda self, job_id: reaped.append(job_id))

        manager.start_server(MODEL, make_config())

        assert reaped == ["555"]
        data = json.loads(stale.read_text())
        assert data["closed"] is True
        assert data["closed_by"] == manager._run_id

    def test_live_pid_left_alone(self, manager, monkeypatch, tmp_path):
        stale = write_stale_ledger(tmp_path, socket.gethostname(), 99999, ["555"])
        submit_recorder(monkeypatch, manager)
        monkeypatch.setattr(csm, "_pid_alive", lambda pid: True)
        reaped = []
        monkeypatch.setattr(
            ClusterModelServerManager, "_scancel",
            lambda self, job_id: reaped.append(job_id))

        manager.start_server(MODEL, make_config())

        assert reaped == []
        assert json.loads(stale.read_text())["closed"] is False

    def test_other_host_reported_not_reaped(self, manager, monkeypatch, tmp_path):
        stale = write_stale_ledger(tmp_path, "some-other-login-node", 99999, ["555"])
        submit_recorder(monkeypatch, manager)
        reaped = []
        monkeypatch.setattr(
            ClusterModelServerManager, "_scancel",
            lambda self, job_id: reaped.append(job_id))

        manager.start_server(MODEL, make_config())

        assert reaped == []
        assert json.loads(stale.read_text())["closed"] is False

    def test_query_failure_leaves_ledger_open(self, manager, monkeypatch, tmp_path):
        stale = write_stale_ledger(tmp_path, socket.gethostname(), 99999, ["555"])
        submit_recorder(monkeypatch, manager)
        monkeypatch.setattr(csm, "_pid_alive", lambda pid: False)
        monkeypatch.setattr(
            ClusterModelServerManager, "_get_job_state",
            lambda self, job_id, config: ClusterModelServerManager.STATE_QUERY_FAILED)

        manager.start_server(MODEL, make_config())

        assert json.loads(stale.read_text())["closed"] is False

    def test_closed_ledger_ignored(self, manager, monkeypatch, tmp_path):
        write_stale_ledger(
            tmp_path, socket.gethostname(), 99999, ["555"], closed=True)
        submit_recorder(monkeypatch, manager)
        monkeypatch.setattr(csm, "_pid_alive", lambda pid: False)
        reaped = []
        monkeypatch.setattr(
            ClusterModelServerManager, "_scancel",
            lambda self, job_id: reaped.append(job_id))

        manager.start_server(MODEL, make_config())
        assert reaped == []


# ==================== T3: graceful-death teardown ====================

class TestTeardownHandlers:
    def _bare_manager(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            ClusterModelServerManager, "_monitor_loop", lambda self: None)
        return ClusterModelServerManager()

    def test_registers_atexit_and_signals_once(self, monkeypatch, tmp_path):
        m = self._bare_manager(monkeypatch, tmp_path)
        atexit_calls, signal_calls = [], []
        monkeypatch.setattr(csm.atexit, "register", atexit_calls.append)
        monkeypatch.setattr(
            csm.signal, "signal",
            lambda sig, handler: signal_calls.append(sig) or "PREV")

        m._register_teardown({})
        m._register_teardown({})  # idempotent

        assert atexit_calls == [m._atexit_teardown]
        assert signal_calls == [signal.SIGTERM, signal.SIGINT]
        assert m._prev_signal_handlers == {
            signal.SIGTERM: "PREV", signal.SIGINT: "PREV"}

    def test_signal_handlers_opt_out(self, monkeypatch, tmp_path):
        m = self._bare_manager(monkeypatch, tmp_path)
        atexit_calls, signal_calls = [], []
        monkeypatch.setattr(csm.atexit, "register", atexit_calls.append)
        monkeypatch.setattr(
            csm.signal, "signal",
            lambda sig, handler: signal_calls.append(sig))

        m._register_teardown({"install_signal_handlers": False})

        assert len(atexit_calls) == 1
        assert signal_calls == []

    def test_signal_teardown_shuts_down_and_chains(self, monkeypatch, tmp_path):
        m = self._bare_manager(monkeypatch, tmp_path)
        shutdowns, chained = [], []
        monkeypatch.setattr(m, "shutdown_all", lambda: shutdowns.append(True))
        m._prev_signal_handlers[signal.SIGTERM] = (
            lambda sig, frame: chained.append(sig))

        m._signal_teardown(signal.SIGTERM, None)

        assert shutdowns == [True]
        assert chained == [signal.SIGTERM]

    def test_signal_teardown_redelivers_on_default(self, monkeypatch, tmp_path):
        m = self._bare_manager(monkeypatch, tmp_path)
        monkeypatch.setattr(m, "shutdown_all", lambda: None)
        m._prev_signal_handlers[signal.SIGTERM] = signal.SIG_DFL
        restored, kills = [], []
        monkeypatch.setattr(
            csm.signal, "signal",
            lambda sig, handler: restored.append((sig, handler)))
        monkeypatch.setattr(
            csm.os, "kill", lambda pid, sig: kills.append((pid, sig)))

        m._signal_teardown(signal.SIGTERM, None)

        assert restored == [(signal.SIGTERM, signal.SIG_DFL)]
        assert kills == [(csm.os.getpid(), signal.SIGTERM)]

    def test_signal_teardown_respects_sig_ign(self, monkeypatch, tmp_path):
        m = self._bare_manager(monkeypatch, tmp_path)
        monkeypatch.setattr(m, "shutdown_all", lambda: None)
        m._prev_signal_handlers[signal.SIGINT] = signal.SIG_IGN
        kills = []
        monkeypatch.setattr(
            csm.os, "kill", lambda pid, sig: kills.append((pid, sig)))

        m._signal_teardown(signal.SIGINT, None)
        assert kills == []

    def test_atexit_teardown_only_fires_with_jobs(self, monkeypatch, tmp_path):
        m = self._bare_manager(monkeypatch, tmp_path)
        shutdowns = []
        monkeypatch.setattr(m, "shutdown_all", lambda: shutdowns.append(True))
        m._atexit_teardown()
        assert shutdowns == []
        m._jobs[MODEL] = [{"job_id": "1"}]
        m._atexit_teardown()
        assert shutdowns == [True]
        m._jobs.clear()


# ==================== T4: hysteresis + recovery ====================

def pool_entry(job_id="2001", endpoint="http://node1:8000/v1"):
    return {
        "endpoint": endpoint,
        "is_available": True,
        "job_id": job_id,
        "consecutive_health_failures": 0,
    }


class TestEvictionHysteresis:
    def _managed(self, manager, threshold=3):
        cfg = make_config(eviction_failure_threshold=threshold)
        manager.model_configs[MODEL] = cfg
        entry = pool_entry()
        manager._pool[MODEL] = [entry]
        manager._jobs[MODEL] = [
            {"job_id": entry["job_id"], "port": 8000, "instance_id": 0,
             "discovered": True}]
        return cfg, entry

    def test_survives_below_threshold(self, manager, monkeypatch):
        cfg, entry = self._managed(manager)
        monkeypatch.setattr(
            ClusterModelServerManager, "_health_check",
            lambda self, endpoint, config: (False, "refused"))
        manager._recheck_pool_health(MODEL, cfg)
        manager._recheck_pool_health(MODEL, cfg)
        assert manager._pool[MODEL] == [entry]
        assert entry["consecutive_health_failures"] == 2

    def test_success_resets_counter(self, manager, monkeypatch):
        cfg, entry = self._managed(manager)
        entry["consecutive_health_failures"] = 2
        monkeypatch.setattr(
            ClusterModelServerManager, "_health_check",
            lambda self, endpoint, config: (True, None))
        manager._recheck_pool_health(MODEL, cfg)
        assert entry["consecutive_health_failures"] == 0
        assert manager._pool[MODEL] == [entry]

    def test_evicts_at_threshold(self, manager, monkeypatch):
        cfg, entry = self._managed(manager)
        monkeypatch.setattr(
            ClusterModelServerManager, "_health_check",
            lambda self, endpoint, config: (False, "refused"))
        for _ in range(3):
            manager._recheck_pool_health(MODEL, cfg)
        assert manager._pool[MODEL] == []
        assert manager._evicted[MODEL] == [entry]

    def test_recovery_readds_while_job_alive(self, manager, monkeypatch):
        cfg, entry = self._managed(manager)
        entry["is_available"] = False  # was held when it went dark
        manager._pool[MODEL] = []
        manager._evicted[MODEL] = [entry]
        monkeypatch.setattr(
            ClusterModelServerManager, "_get_job_state",
            lambda self, job_id, config: "RUNNING")
        monkeypatch.setattr(
            ClusterModelServerManager, "_health_check",
            lambda self, endpoint, config: (True, None))

        manager._reprobe_evicted(MODEL, cfg)

        assert manager._evicted[MODEL] == []
        assert manager._pool[MODEL] == [entry]
        assert entry["is_available"] is True
        assert entry["consecutive_health_failures"] == 0

    def test_evicted_dropped_once_job_gone(self, manager, monkeypatch):
        cfg, entry = self._managed(manager)
        manager._pool[MODEL] = []
        manager._evicted[MODEL] = [entry]
        monkeypatch.setattr(
            ClusterModelServerManager, "_get_job_state",
            lambda self, job_id, config: None)

        manager._reprobe_evicted(MODEL, cfg)

        assert manager._evicted[MODEL] == []
        assert manager._pool[MODEL] == []

    def test_still_unhealthy_stays_evicted(self, manager, monkeypatch):
        cfg, entry = self._managed(manager)
        manager._pool[MODEL] = []
        manager._evicted[MODEL] = [entry]
        monkeypatch.setattr(
            ClusterModelServerManager, "_get_job_state",
            lambda self, job_id, config: "RUNNING")
        monkeypatch.setattr(
            ClusterModelServerManager, "_health_check",
            lambda self, endpoint, config: (False, "refused"))

        manager._reprobe_evicted(MODEL, cfg)

        assert manager._evicted[MODEL] == [entry]
        assert manager._pool[MODEL] == []

    def test_acquire_waits_for_recovery_not_all_failed(self, manager):
        """Evicted-but-recovering endpoints must NOT trip the all-failed
        short-circuit; the caller gets the plain timeout instead."""
        cfg, entry = self._managed(manager)
        manager.model_configs[MODEL]["cluster_server_endpoint_timeout"] = 0.2
        manager._pool[MODEL] = []
        manager._evicted[MODEL] = [entry]
        with pytest.raises(RuntimeError, match="No endpoint available"):
            manager.acquire_endpoint(MODEL)

    def test_acquire_all_failed_without_evicted(self, manager):
        cfg, entry = self._managed(manager)
        manager._pool[MODEL] = []
        with pytest.raises(RuntimeError, match="failed during discovery"):
            manager.acquire_endpoint(MODEL, timeout=5)


# ==================== T5: per-model SLURM timeouts ====================

class TestPerModelTimeouts:
    def test_job_state_uses_each_models_timeout(self, manager, monkeypatch):
        seen = []

        class FakeResult:
            returncode = 0
            stdout = "RUNNING"
            stderr = ""

        def fake_run(cmd, capture_output=True, text=True, timeout=None):
            seen.append(timeout)
            return FakeResult()

        monkeypatch.setattr(csm.subprocess, "run", fake_run)
        manager._get_job_state("1", make_config(slurm_cmd_timeout=7))
        manager._get_job_state("2", make_config(slurm_cmd_timeout=23))
        assert seen == [7, 23]

    def test_health_check_timeout_from_own_config(self, manager, monkeypatch):
        seen = {}

        class FakeOpener:
            def open(self, req, timeout=None):
                seen["timeout"] = timeout
                raise OSError("no server in tests")

        monkeypatch.setattr(
            "urllib.request.build_opener", lambda *a, **k: FakeOpener())
        healthy, err = manager._health_check(
            "http://node1:8000/v1", make_config(health_check_timeout=42))
        assert healthy is False
        assert seen["timeout"] == 42


# ==================== Condition-based pool ====================

class TestConditionPool:
    def test_release_wakes_blocked_acquire(self, manager):
        entry = pool_entry()
        entry["is_available"] = False
        manager.model_configs[MODEL] = make_config()
        manager._pool[MODEL] = [entry]
        manager._jobs[MODEL] = [
            {"job_id": entry["job_id"], "port": 8000, "instance_id": 0,
             "discovered": True}]

        got = []
        t = threading.Thread(
            target=lambda: got.append(manager.acquire_endpoint(MODEL, timeout=5)))
        t.start()
        manager.release_endpoint(MODEL, entry["endpoint"])
        t.join(timeout=3)

        assert not t.is_alive()
        assert got == [entry["endpoint"]]
        assert entry["is_available"] is False  # re-acquired by the waiter

    def test_acquire_immediate_when_available(self, manager):
        entry = pool_entry()
        manager.model_configs[MODEL] = make_config()
        manager._pool[MODEL] = [entry]
        manager._jobs[MODEL] = [
            {"job_id": entry["job_id"], "port": 8000, "instance_id": 0,
             "discovered": True}]
        assert manager.acquire_endpoint(MODEL) == entry["endpoint"]
        assert entry["is_available"] is False

    def test_status_reports_evicted(self, manager):
        entry = pool_entry()
        manager.model_configs[MODEL] = make_config()
        manager._pool[MODEL] = []
        manager._evicted[MODEL] = [entry]
        manager._jobs[MODEL] = [
            {"job_id": entry["job_id"], "port": 8000, "instance_id": 0,
             "discovered": True}]
        status = manager.get_server_status(MODEL)
        assert status["num_evicted"] == 1
        assert status["num_ready"] == 0


# ==================== sbatch submission retry ====================

class TestSubmitRetry:
    def _fake_runs(self, monkeypatch, outcomes):
        """outcomes: list of (returncode, stdout, stderr) per attempt."""
        calls = []

        class R:
            def __init__(self, rc, out, err):
                self.returncode, self.stdout, self.stderr = rc, out, err

        def fake_run(cmd, capture_output=True, text=True, timeout=None):
            calls.append(cmd)
            rc, out, err = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
            return R(rc, out, err)

        monkeypatch.setattr(csm.subprocess, "run", fake_run)
        monkeypatch.setattr(csm.time, "sleep", lambda s: None)
        return calls

    def test_transient_nonzero_exit_retried(self, manager, monkeypatch, tmp_path):
        calls = self._fake_runs(monkeypatch, [
            (1, "", "sbatch: error: Socket timed out on send/recv operation"),
            (1, "", "sbatch: error: Socket timed out on send/recv operation"),
            (0, "Submitted batch job 777", ""),
        ])
        script = tmp_path / "x.sbatch"
        script.write_text("#!/bin/bash\n")
        assert manager._submit_sbatch(script) == "777"
        assert len(calls) == 3

    def test_gives_up_after_four_attempts(self, manager, monkeypatch, tmp_path):
        calls = self._fake_runs(monkeypatch, [
            (1, "", "sbatch: error: invalid partition specified"),
        ])
        script = tmp_path / "x.sbatch"
        script.write_text("#!/bin/bash\n")
        with pytest.raises(RuntimeError, match="invalid partition"):
            manager._submit_sbatch(script)
        assert len(calls) == 4


# ==================== run-scoped log dir ====================

class TestLogDir:
    def test_default_log_dir_is_logs(self, manager):
        manager._resolve_log_dir(make_config())
        assert str(manager._log_dir) == "logs"
        assert str(manager._ledger_dir) == "logs"

    def test_run_scoped_logs_nest_but_ledger_stays_base(self, manager):
        manager._resolve_log_dir(make_config(run_scoped_logs=True))
        assert str(manager._log_dir) == f"logs/run_{manager._run_id}"
        assert str(manager._ledger_dir) == "logs"

    def test_sbatch_paths_use_run_scoped_dir(self, manager, tmp_path):
        cfg = make_config(run_scoped_logs=True)
        path = manager._generate_sbatch(MODEL, cfg, instance_id=0)
        script = path.read_text()
        assert f"--output=logs/run_{manager._run_id}/" in script
        assert f"--error=logs/run_{manager._run_id}/" in script
        assert (tmp_path / "logs" / f"run_{manager._run_id}").is_dir()
