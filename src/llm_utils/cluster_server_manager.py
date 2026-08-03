"""
Manages vLLM server lifecycle on SLURM cluster.

Handles:
- Auto-generating sbatch scripts for vLLM servers
- Submitting SLURM jobs and tracking their IDs
- Discovering endpoints via scontrol (no shared files needed)
- Multi-instance support: N servers per model on different ports
- Dynamic server pool with acquire/release endpoint allocation
- Background monitoring: servers are added to the pool as they become healthy
- Zombie prevention: durable on-disk job ledger + atexit/SIGTERM/SIGINT teardown
- Health-check hysteresis: N consecutive failures to evict; evicted servers
  whose SLURM job still runs are re-probed and re-added on recovery
"""
import atexit
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_model import LLMModel, Provider
from ._logging import get_logger

logger = get_logger(__name__)

ClusterConfigDict = Dict[str, Any]

# Config keys every server instance needs. Checked for ALL instances BEFORE
# the first sbatch, so a bad config never dies mid-submission with jobs
# already up (family cluster-job standard, rule T1).
REQUIRED_CONFIG_KEYS = (
    "num_instances",
    "port",
    "partition",
    "gpu_memory_utilization",
    "max_model_len",
    "num_gpus",
    "cpus_per_task",
    "mem_gb",
    "time_limit",
    "hf_home",
)
# Required only when config['env_setup'] does not replace the default
# conda-module env lines.
CONDA_ENV_KEYS = ("cuda_module", "conda_env")


def _pid_alive(pid: int) -> bool:
    """True if a process with this pid exists on THIS host (signal-0 probe)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # PermissionError etc.: the process exists but isn't ours.
        return True
    return True


def _slurm_time_to_seconds(t: Any) -> Optional[int]:
    """Parse a SLURM time spec to seconds. Accepts ``D-HH:MM:SS``,
    ``D-HH:MM``, ``D-HH``, ``HH:MM:SS``, ``MM:SS``, bare minutes (``"90"``),
    or an int (minutes) — sbatch's own accepted forms. Returns None if
    unparseable."""
    if isinstance(t, int):
        return t * 60
    s = str(t).strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        try:
            days = int(d)
        except ValueError:
            return None
    try:
        nums = [int(p) for p in s.split(":")]
    except ValueError:
        return None
    if "-" in str(t):
        # After a day part: HH[:MM[:SS]]
        if len(nums) == 1:
            h, m, sec = nums[0], 0, 0
        elif len(nums) == 2:
            h, m, sec = nums[0], nums[1], 0
        elif len(nums) == 3:
            h, m, sec = nums
        else:
            return None
    else:
        if len(nums) == 1:          # bare minutes
            h, m, sec = 0, nums[0], 0
        elif len(nums) == 2:        # MM:SS
            h, m, sec = 0, nums[0], nums[1]
        elif len(nums) == 3:        # HH:MM:SS
            h, m, sec = nums
        else:
            return None
    return ((days * 24 + h) * 60 + m) * 60 + sec


class ClusterModelServerManager:
    """
    Manages vLLM server lifecycle on SLURM cluster with dynamic endpoint pool.

    Supports multiple server instances per model for parallel task execution.
    Each instance gets its own SLURM job, GPU, and port.

    Servers are added to the pool dynamically as they become healthy.
    Tasks acquire/release endpoints — no blocking on all servers at startup.

    Zombie prevention: every submitted job id is persisted to an on-disk
    ledger (``<log_dir>/slurm_jobs_<run_id>.json``) at submission time, so
    even a SIGKILL'd orchestrator leaves a reap list — the next run from the
    same directory reaps provably-orphaned jobs (same host, owner pid dead)
    and reports the rest. Graceful deaths (atexit, SIGTERM, SIGINT) scancel
    owned jobs directly.

    Optional config keys (fail-safe defaults in code; consumers tune them in
    their cluster YAML):
        log_dir ("logs")               — sbatch stdout/stderr + ledger dir
        run_scoped_logs (False)        — nest job logs in a per-run subdir so
                                         concurrent runs don't interleave
        reap_orphans (True)            — False = report orphans, never scancel
        install_signal_handlers (True) — False = atexit teardown only
        eviction_failure_threshold (3) — consecutive health failures to evict
                                         (tuning data: the EVICTION log lines)

    Usage:
        manager = ClusterModelServerManager()
        manager.start_server(LLMModel.PIXTRAL_12B, config)  # returns immediately

        endpoint = manager.acquire_endpoint(model)  # blocks until one is available
        # ... use endpoint ...
        manager.release_endpoint(model, endpoint)

        manager.shutdown_all()
    """

    def __init__(self):
        self._jobs: Dict[LLMModel, List[dict]] = {}
        self._pool: Dict[LLMModel, List[dict]] = {}
        # One Condition guards _pool AND _evicted. A real Condition (not the
        # old check-then-Event pattern, which could miss a wakeup slipped in
        # between a waiter's check and its wait under concurrent
        # acquire/release).
        self._pool_cond = threading.Condition()
        # Endpoints evicted by health hysteresis, keyed by model — kept for
        # recovery re-probing while their SLURM job is still alive.
        self._evicted: Dict[LLMModel, List[dict]] = {}

        self.model_configs: Dict[LLMModel, ClusterConfigDict] = {}

        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()

        self._sbatch_dir = Path(tempfile.mkdtemp(prefix="vllm_sbatch_"))

        # Cache: (partition, frozenset(excluded_types)) -> [node names with excluded GPU]
        self._gpu_type_exclude_cache: Dict[tuple, List[str]] = {}

        # Durable job ledger: every submitted job id is persisted at
        # submission time so ANY orchestrator death — including SIGKILL,
        # which no handler catches — leaves a reap list for the next run.
        self._run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        self._ledger_jobs: List[dict] = []
        self._ledger_closed = False
        self._log_dir: Optional[Path] = None     # resolved at first start_server
        self._ledger_dir: Optional[Path] = None  # always the BASE log dir, so
                                                 # later runs can discover
                                                 # prior ledgers even when job
                                                 # logs are run-scoped
        self._orphans_checked = False

        # Graceful-death teardown, registered once at first submission.
        self._teardown_registered = False
        self._prev_signal_handlers: Dict[int, Any] = {}

    # ==================== Public API ====================

    def start_server(self, model: LLMModel, config: ClusterConfigDict) -> None:
        """
        Start vLLM server(s) for the given model.

        Validates the full config and generates every instance's sbatch
        script BEFORE the first submission, then submits SLURM jobs and
        starts a background monitor thread. Returns immediately — does NOT
        wait for servers to be healthy.

        Args:
            model: The cluster model to serve.
            config: Cluster config dict (the consumer supplies it — this
                package has no config-file knowledge).
        """
        if model in self._jobs and self._jobs[model]:
            logger.info(f"Servers already submitted for {model.model_id}")
            return

        if model.provider != Provider.SLURM_CLUSTER:
            raise ValueError(
                f"{model.model_id} is not a cluster model (provider: {model.provider})")

        self._validate_config(model, config)

        self.model_configs[model] = config
        self._resolve_log_dir(config)
        if not self._orphans_checked:
            self._orphans_checked = True
            try:
                self._reap_orphan_ledgers(config)
            except Exception:
                logger.exception("orphan-ledger check failed — continuing")

        num_instances = config["num_instances"]
        base_port = config["port"]

        logger.info(
            f"Starting {num_instances} vLLM server(s) for {model.model_id} "
            f"(ports {base_port}-{base_port + num_instances - 1})")

        # Generate every instance's script before submitting any of them:
        # generation errors (missing chat template, bad time spec) surface
        # with zero jobs up. A submission failure mid-batch can still leave
        # earlier instances running — those are already in the ledger, and
        # the atexit/signal handlers reap them if the run dies on the error.
        sbatch_paths: List[Path] = []
        for i in range(num_instances):
            instance_config = {**config, "port": base_port + i}
            sbatch_paths.append(
                self._generate_sbatch(model, instance_config, instance_id=i))

        self._register_teardown(config)

        self._jobs[model] = []
        with self._pool_cond:
            self._pool[model] = []

        for i, sbatch_path in enumerate(sbatch_paths):
            instance_port = base_port + i
            job_id = self._submit_sbatch(sbatch_path)

            self._jobs[model].append({
                "job_id": job_id,
                "port": instance_port,
                "instance_id": i,
                "discovered": False,
            })
            self._ledger_jobs.append({
                "job_id": job_id,
                "model": model.name,
                "port": instance_port,
                "instance_id": i,
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "cancelled": False,
            })
            self._write_ledger()
            logger.info(f"  Instance {i}: SLURM job {job_id}, port {instance_port}")

        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_monitor.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="vllm-pool-monitor",
            )
            self._monitor_thread.start()
            logger.info("Started background server monitor thread")

    def acquire_endpoint(self, model: LLMModel, timeout: Optional[int] = None) -> str:
        """
        Acquire an available endpoint from the pool. Blocks until one is available.

        Returns:
            Endpoint URL string.

        Raises:
            RuntimeError: If no endpoint becomes available within timeout.
        """
        config = self.model_configs.get(model, {})
        wait_interval = config.get("endpoint_wait_timeout", 30)
        if timeout is None:
            timeout = config.get("cluster_server_endpoint_timeout", 10000)
        deadline = time.time() + timeout

        with self._pool_cond:
            while True:
                for entry in self._pool.get(model, []):
                    if entry["is_available"]:
                        entry["is_available"] = False
                        logger.debug(f"Acquired endpoint {entry['endpoint']}")
                        return entry["endpoint"]

                # Fail fast if NO server was ever submitted for this model — a
                # serve-discovery miss (a model the task requested that the
                # orchestrator never added to its cluster-model set). Without this,
                # acquire blocks the full cluster_server_endpoint_timeout (~2.8h)
                # on a pool that can never fill (observed 2026-07-16).
                jobs = self._jobs.get(model, [])
                if not jobs:
                    raise RuntimeError(
                        f"No vLLM server was ever started for {model.model_id}: a task "
                        f"requested it (e.g. as a judge/guard) but the orchestrator did "
                        f"not add it to its cluster-model set (serve-discovery miss). "
                        f"Check the orchestrator's serve wiring.")
                # Short-circuit on all-failed: if every submitted job for this
                # model is already discovered (resolved by monitor) and the pool
                # is still empty, every one of them failed — no point waiting.
                # An evicted-but-recovering endpoint (its job still RUNNING)
                # does NOT count as failed: keep waiting for the monitor's
                # recovery re-probe instead.
                if (all(j["discovered"] for j in jobs)
                        and not self._pool.get(model)
                        and not self._evicted.get(model)):
                    raise RuntimeError(
                        f"All {len(jobs)} server job(s) for "
                        f"{model.model_id} failed during discovery. "
                        f"Check {self._err_log_hint(model)} for "
                        f"startup errors (config mismatch, QoS-blocked, OOM, etc.).")

                remaining = deadline - time.time()
                if remaining <= 0:
                    pool_size = len(self._pool.get(model, []))
                    break
                self._pool_cond.wait(timeout=min(wait_interval, remaining))

        raise RuntimeError(
            f"No endpoint available for {model.model_id} within {timeout}s. "
            f"Pool has {pool_size} server(s), "
            f"all busy or none started.")

    def release_endpoint(self, model: LLMModel, endpoint: str) -> None:
        """Release an endpoint back to the pool, marking it as available."""
        with self._pool_cond:
            for entry in self._pool.get(model, []):
                if entry["endpoint"] == endpoint:
                    entry["is_available"] = True
                    logger.debug(f"Released endpoint {endpoint}")
                    break
            self._pool_cond.notify_all()

    def get_num_instances(self, model: LLMModel) -> int:
        """Get the number of submitted server instances for a model."""
        return len(self._jobs.get(model, []))

    def get_num_ready(self, model: LLMModel) -> int:
        """Get the number of healthy, pool-registered endpoints for a model."""
        with self._pool_cond:
            return len(self._pool.get(model, []))

    def wait_for_first_server(self, model: LLMModel, timeout: Optional[int] = None) -> str:
        """Block until at least one server for `model` enters the pool.

        Short-circuits if all submitted server jobs for this model have
        already been resolved by the monitor and none added to the pool —
        i.e., every job failed. Without this, we'd keep polling an empty
        pool until full timeout (1h default), even though the manager
        already knows the model has no live servers.

        Returns:
            The first available endpoint URL.

        Raises:
            RuntimeError: If no server comes up within timeout, OR if the
                monitor has confirmed every job for this model failed.
        """
        config = self.model_configs.get(model, {})
        if timeout is None:
            timeout = config.get("server_start_timeout", 3600)
        deadline = time.time() + timeout
        start = time.time()
        logger.info(f"Waiting for first server for {model.model_id} (timeout: {timeout}s)...")

        last_progress_log = start
        progress_interval = 60.0  # emit a "still waiting" status every 60s

        with self._pool_cond:
            while True:
                if self._pool.get(model):
                    first_endpoint = self._pool[model][0]["endpoint"]
                    logger.info(f"First server ready: {first_endpoint}")
                    return first_endpoint

                # Fail fast if NO server was ever submitted for this model
                # (serve-discovery miss) — never block the full timeout on a pool
                # that can never fill. Parallels acquire_endpoint's guard.
                jobs = self._jobs.get(model, [])
                if not jobs:
                    raise RuntimeError(
                        f"No vLLM server was ever started for {model.model_id} "
                        f"(serve-discovery miss). Check the orchestrator's serve "
                        f"wiring.")
                # Short-circuit on all-failed (see acquire_endpoint; evicted
                # entries pending recovery don't count as failed).
                if (all(j["discovered"] for j in jobs)
                        and not self._pool.get(model)
                        and not self._evicted.get(model)):
                    raise RuntimeError(
                        f"All {len(jobs)} server job(s) for "
                        f"{model.model_id} failed during discovery. "
                        f"Check {self._err_log_hint(model)} for "
                        f"startup errors (config mismatch, OOM, etc.).")

                # Periodic progress log so an extended wait isn't silent (the
                # monitor only logs successes/failures; PENDING + slow-server
                # would otherwise produce no orchestrator-side output for many
                # minutes).
                now = time.time()
                if now - last_progress_log >= progress_interval:
                    pool_size = len(self._pool.get(model, []))
                    discovered = sum(1 for j in jobs if j["discovered"])
                    elapsed = int(now - start)
                    logger.info(
                        f"Still waiting for {model.model_id}: elapsed={elapsed}s, "
                        f"jobs_submitted={len(jobs)}, jobs_discovered="
                        f"{discovered}, pool_size={pool_size}. "
                        f"(See per-job monitor lines above for what's blocking.)")
                    last_progress_log = now

                remaining = deadline - now
                if remaining <= 0:
                    break
                self._pool_cond.wait(timeout=min(10, remaining))

        raise RuntimeError(
            f"No server for {model.model_id} became ready within {timeout}s. "
            f"Check SLURM logs.")

    def get_server_status(self, model: LLMModel) -> Dict[str, Any]:
        """Get the current lifecycle status of servers for a model."""
        status: Dict[str, Any] = {
            "state": "not_started",
            "job_ids": [],
            "endpoints": [],
            "num_submitted": 0,
            "num_ready": 0,
            "num_available": 0,
            "num_evicted": 0,
        }

        if model not in self._jobs:
            return status

        jobs = self._jobs[model]
        status["job_ids"] = [j["job_id"] for j in jobs]
        status["num_submitted"] = len(jobs)

        with self._pool_cond:
            pool = self._pool.get(model, [])
            status["num_ready"] = len(pool)
            status["num_available"] = sum(1 for e in pool if e["is_available"])
            status["endpoints"] = [e["endpoint"] for e in pool]
            status["num_evicted"] = len(self._evicted.get(model, []))

        if status["num_ready"] == 0:
            status["state"] = "pending"
        elif status["num_ready"] < status["num_submitted"]:
            status["state"] = "partially_ready"
        else:
            status["state"] = "ready"

        return status

    def shutdown_all(self) -> None:
        """Cancel all active SLURM jobs and clean up.

        Idempotent — also runs from atexit and the SIGTERM/SIGINT handlers,
        possibly after a manual call already did the work.
        """
        self._stop_monitor.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

        if not self._jobs and self._ledger_closed:
            return  # already shut down (e.g. atexit after a manual call)

        cancelled_ids: List[str] = []
        for model, jobs in self._jobs.items():
            for job_info in jobs:
                job_id = job_info["job_id"]
                logger.info(f"Cancelling SLURM job {job_id} ({model.model_id})")
                self._scancel(job_id)
                cancelled_ids.append(job_id)

        self._jobs.clear()
        with self._pool_cond:
            self._pool.clear()
            self._evicted.clear()
            self._pool_cond.notify_all()

        # A closed ledger tells the next run there is nothing to reap.
        if cancelled_ids:
            self._mark_ledger_cancelled(cancelled_ids)
        self._ledger_closed = True
        self._write_ledger()
        logger.info("All vLLM servers shut down")

    def shutdown_model(self, model: LLMModel) -> None:
        """Cancel all SLURM jobs for a single model and clear its pool entries.

        Mid-run teardown: lets the orchestrator free a heavy model's GPU
        allocation as soon as the last task needing it completes, instead of
        holding 4× A100s pinned for the entire experiment lifetime.

        Safe to call on a model that's already been torn down (no-op).
        """
        jobs = self._jobs.pop(model, [])
        cancelled_ids: List[str] = []
        for job_info in jobs:
            job_id = job_info["job_id"]
            logger.info(
                f"Cancelling SLURM job {job_id} ({model.model_id}) [mid-run]")
            self._scancel(job_id)
            cancelled_ids.append(job_id)
        with self._pool_cond:
            self._pool.pop(model, None)
            self._evicted.pop(model, None)
            self._pool_cond.notify_all()
        if cancelled_ids:
            self._mark_ledger_cancelled(cancelled_ids)

    def __del__(self):
        if self._jobs:
            self.shutdown_all()

    # ==================== Config Validation ====================

    def _validate_config(self, model: LLMModel, config: ClusterConfigDict) -> None:
        """Validate every required key BEFORE the first sbatch (T1) — a bad
        config must never die mid-submission with jobs already up. Reports
        ALL missing keys at once, not the first KeyError hit."""
        missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
        if not config.get("env_setup"):
            missing += [k for k in CONDA_ENV_KEYS if k not in config]
        if missing:
            raise ValueError(
                f"Cluster config for {model.model_id} is missing required "
                f"key(s): {', '.join(sorted(missing))}. Nothing was submitted.")
        n = config["num_instances"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(
                f"num_instances for {model.model_id} must be a positive int, "
                f"got {n!r}. Nothing was submitted.")

    # ==================== Log Dir & Job Ledger ====================

    def _resolve_log_dir(self, config: ClusterConfigDict) -> Path:
        """Resolve the log directory once, at first start_server.

        ``log_dir`` (default "logs") is where sbatch stdout/stderr land;
        ``run_scoped_logs: true`` nests them in a per-run subdir so
        concurrent runs from one CWD don't interleave. Job ledgers always
        stay in the BASE dir so later runs can discover prior runs' ledgers.
        """
        if self._log_dir is None:
            base = Path(config.get("log_dir", "logs"))
            self._ledger_dir = base
            if config.get("run_scoped_logs"):
                self._log_dir = base / f"run_{self._run_id}"
            else:
                self._log_dir = base
        return self._log_dir

    def _err_log_hint(self, model: LLMModel) -> str:
        """Where to look for a failed server's stderr, for error messages."""
        return f"{self._log_dir or 'logs'}/vllm_{model.name.lower()}_*.err"

    def _ledger_path(self) -> Path:
        return self._ledger_dir / f"slurm_jobs_{self._run_id}.json"

    def _write_ledger(self) -> None:
        """Persist the job ledger (atomic tmp+rename). Best-effort: a ledger
        IO failure must not kill the run — it only degrades crash reaping."""
        if self._ledger_dir is None:
            return
        data = {
            "run_id": self._run_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "closed": self._ledger_closed,
            "jobs": self._ledger_jobs,
        }
        try:
            self._ledger_dir.mkdir(parents=True, exist_ok=True)
            path = self._ledger_path()
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, path)
        except OSError as e:
            logger.warning(f"could not write job ledger {self._ledger_path()}: {e}")

    def _mark_ledger_cancelled(self, job_ids: List[str]) -> None:
        ids = set(job_ids)
        for entry in self._ledger_jobs:
            if entry["job_id"] in ids:
                entry["cancelled"] = True
        self._write_ledger()

    def _reap_orphan_ledgers(self, config: ClusterConfigDict) -> None:
        """Reap-or-report jobs from PRIOR runs' unclosed ledgers (T2).

        Reaps (scancel) only when the owning orchestrator is provably dead:
        same host AND its pid no longer exists. A live pid means a concurrent
        run from this CWD — leave its servers alone. A different host means
        the pid can't be probed from here — report, never guess. Set config
        ``reap_orphans: false`` to report-only.
        """
        if self._ledger_dir is None or not self._ledger_dir.exists():
            return
        reap = config.get("reap_orphans", True)
        this_host = socket.gethostname()

        for path in sorted(self._ledger_dir.glob("slurm_jobs_*.json")):
            if path.name == self._ledger_path().name:
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"unreadable job ledger {path}: {e}")
                continue
            if data.get("closed"):
                continue
            open_jobs = [j for j in data.get("jobs", []) if not j.get("cancelled")]
            if not open_jobs:
                continue

            host, pid = data.get("host"), data.get("pid")
            job_ids = [str(j["job_id"]) for j in open_jobs]
            if host != this_host:
                logger.warning(
                    f"Unclosed job ledger {path.name} from host {host!r} "
                    f"(this host: {this_host!r}) — cannot verify its "
                    f"orchestrator from here. If that run crashed, scancel "
                    f"manually: scancel {' '.join(job_ids)}")
                continue
            try:
                owner_alive = bool(pid) and _pid_alive(int(pid))
            except (TypeError, ValueError):
                owner_alive = False
            if owner_alive:
                logger.info(
                    f"Job ledger {path.name} belongs to live pid {pid} "
                    f"(likely a concurrent run from this directory) — "
                    f"leaving its jobs alone.")
                continue

            # Provably orphaned: same host, owner pid dead.
            if not reap:
                logger.warning(
                    f"Orphaned job ledger {path.name} (dead pid {pid}); "
                    f"reap_orphans=false — scancel manually: "
                    f"scancel {' '.join(job_ids)}")
                continue

            all_verified = True
            for j in open_jobs:
                job_id = str(j["job_id"])
                state = self._get_job_state(job_id, config)
                if state == self.STATE_QUERY_FAILED:
                    # Leave the ledger open so the NEXT run retries this job.
                    all_verified = False
                    logger.warning(
                        f"could not verify orphan job {job_id} (squeue "
                        f"failed); leaving ledger {path.name} open")
                    continue
                if state is None:
                    continue  # already left the queue
                logger.warning(
                    f"Reaping orphaned SLURM job {job_id} "
                    f"({j.get('model', '?')}) from dead run "
                    f"{data.get('run_id', path.name)}")
                self._scancel(job_id)
            if all_verified:
                data["closed"] = True
                data["closed_by"] = self._run_id
                try:
                    path.write_text(json.dumps(data, indent=2))
                except OSError as e:
                    logger.warning(f"could not mark ledger {path} closed: {e}")

    # ==================== Graceful-Death Teardown ====================

    def _register_teardown(self, config: ClusterConfigDict) -> None:
        """atexit + SIGTERM/SIGINT teardown (T3): ``__del__`` alone is not
        guaranteed by Python, so graceful deaths must scancel owned jobs
        explicitly. Only SIGKILL escapes — that's what the on-disk ledger
        covers. Handlers chain to whatever was installed before us."""
        if self._teardown_registered:
            return
        self._teardown_registered = True
        atexit.register(self._atexit_teardown)
        if not config.get("install_signal_handlers", True):
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._prev_signal_handlers[sig] = signal.signal(
                    sig, self._signal_teardown)
            except ValueError:
                # Not in the main thread — signal handlers can't be installed
                # from here; atexit still covers normal interpreter exit.
                logger.warning(
                    "not in main thread: SIGTERM/SIGINT teardown handlers not "
                    "installed (atexit teardown still active)")
                return

    def _atexit_teardown(self) -> None:
        if self._jobs:
            logger.info("atexit: tearing down vLLM server jobs")
            self.shutdown_all()

    def _signal_teardown(self, signum, frame) -> None:
        logger.warning(
            f"received signal {signum}: cancelling vLLM server jobs before exit")
        self.shutdown_all()
        prev = self._prev_signal_handlers.get(signum)
        if prev is signal.SIG_IGN:
            return
        if callable(prev):
            # e.g. default_int_handler → KeyboardInterrupt propagates.
            prev(signum, frame)
            return
        # SIG_DFL / None: restore the default disposition and re-deliver so
        # the process exits with the correct signal status.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    # ==================== Background Monitor ====================

    def _monitor_loop(self) -> None:
        """Background thread: discover new servers AND keep checking pool health.

        Phase 1 (discovery): poll SLURM jobs and add healthy endpoints to the
        pool as they become reachable. Each job is "discovered" exactly once.

        Phase 2 (maintenance): per model, on that model's OWN
        ``health_recheck_interval`` cadence (per-model config — T5):
        re-ping every pool entry's /health with eviction hysteresis, then
        re-probe evicted entries whose SLURM job is still alive and re-add
        them on recovery (T4). Without maintenance, acquire_endpoint() would
        hand out a stale dead endpoint and the task would hang on the first
        request rather than waiting for a fresh one.

        Runs until _stop_monitor is set (i.e. until shutdown_all()).
        """
        last_recheck: Dict[LLMModel, float] = {}

        while not self._stop_monitor.is_set():
            # Both phases are exception-guarded: an uncaught error would kill
            # the monitor thread permanently — orphaning every undiscovered
            # job and ending pool health-checks for the rest of the run.
            # ---- Phase 1: discovery ----
            try:
                self._discovery_pass()
            except Exception:
                logger.exception("monitor discovery pass failed — continuing")

            # ---- Phase 2: re-health-check + recovery, per model ----
            try:
                now = time.time()
                for model, config in list(self.model_configs.items()):
                    interval = config.get("health_recheck_interval", 60)
                    if now - last_recheck.get(model, 0.0) < interval:
                        continue
                    last_recheck[model] = now
                    self._recheck_pool_health(model, config)
                    self._reprobe_evicted(model, config)
            except Exception:
                logger.exception("monitor health recheck failed — continuing")

            # Poll on the fastest cadence any served model asked for.
            poll_interval = min(
                (c.get("monitor_poll_interval", 10)
                 for c in self.model_configs.values()),
                default=10)
            self._stop_monitor.wait(timeout=poll_interval)

    def _discovery_pass(self) -> None:
        """One discovery sweep over undiscovered jobs (see _monitor_loop)."""
        for model, jobs in list(self._jobs.items()):
            config = self.model_configs.get(model, {})
            for job_info in jobs:
                if job_info["discovered"]:
                    continue

                job_id = job_info["job_id"]
                port = job_info["port"]
                instance_id = job_info["instance_id"]
                instance_suffix = f"[{instance_id}]" if instance_id > 0 else ""

                state = self._get_job_state(job_id, config)

                if state == self.STATE_QUERY_FAILED:
                    # Transient squeue hiccup (the controller is known to
                    # be intermittently slow) — leave the job undiscovered
                    # and re-examine next poll instead of blackholing a
                    # live server.
                    fails = job_info.get("_state_query_failures", 0) + 1
                    job_info["_state_query_failures"] = fails
                    if fails <= 3 or fails % 30 == 0:
                        logger.info(
                            f"squeue query failed for job {job_id} "
                            f"(attempt {fails}); will retry")
                    continue
                job_info["_state_query_failures"] = 0

                if state is None:
                    logger.warning(
                        f"Server{instance_suffix} (job {job_id}) failed. "
                        f"Check {self._err_log_hint(model)}")
                    job_info["discovered"] = True
                    continue

                if state == "PENDING":
                    # Log first observation only, then quietly poll.
                    if not job_info.get("_logged_pending"):
                        logger.info(
                            f"Server{instance_suffix} (job {job_id}) "
                            f"still PENDING in SLURM queue; will keep polling.")
                        job_info["_logged_pending"] = True
                    continue

                node = self._resolve_job_node(job_id, config)
                if not node:
                    # Log first N occurrences so we know SLURM isn't surfacing
                    # the node assignment yet — silent-forever before this fix.
                    attempts = job_info.get("_node_resolve_attempts", 0) + 1
                    job_info["_node_resolve_attempts"] = attempts
                    if attempts <= 3 or attempts % 30 == 0:
                        logger.info(
                            f"Server{instance_suffix} (job {job_id}) state="
                            f"{state} but scontrol hasn't returned a NodeList/"
                            f"NodeAddr yet (attempt {attempts}). Retrying.")
                    continue

                endpoint = f"http://{node}:{port}/v1"
                healthy, err = self._health_check(endpoint, config)

                if not healthy:
                    # The previously-silent failure mode. Could be:
                    #   - vLLM still loading weights / compiling graphs
                    #   - inter-partition network unreachable (firewall, etc.)
                    #   - wrong NodeAddr resolved
                    #   - vLLM crashed but SLURM hasn't noticed
                    # Log first few + every Nth so user sees something.
                    attempts = job_info.get("_health_check_attempts", 0) + 1
                    job_info["_health_check_attempts"] = attempts
                    if attempts <= 3 or attempts % 12 == 0:  # first 3, then ~every 2min
                        logger.info(
                            f"Server{instance_suffix} (job {job_id}) at "
                            f"{endpoint} not yet healthy "
                            f"(attempt {attempts}, err={err}). Retrying.")

                if healthy:
                    with self._pool_cond:
                        # Guard the teardown race: if this model was shut
                        # down (or shutdown_all cleared the pool) while we
                        # were health-checking, do NOT resurrect an
                        # endpoint for a cancelled job.
                        if (model not in self._pool
                                or self._stop_monitor.is_set()):
                            continue
                        self._pool[model].append({
                            "endpoint": endpoint,
                            "is_available": True,
                            "job_id": job_id,
                            "consecutive_health_failures": 0,
                        })
                        pool_size = len(self._pool[model])
                        self._pool_cond.notify_all()
                    job_info["discovered"] = True
                    logger.info(
                        f"Server{instance_suffix} ready at {endpoint} "
                        f"(pool size: {pool_size})")

    def _recheck_pool_health(
        self, model: LLMModel, config: ClusterConfigDict
    ) -> None:
        """Re-ping every pool entry's /health; evict after N consecutive fails.

        Catches mid-run vLLM crashes (OOM, GPU error, network blip) so that
        acquire_endpoint() doesn't hand out a dead URL. Eviction has
        HYSTERESIS (T4): a single failed ping no longer evicts —
        ``eviction_failure_threshold`` consecutive failures (fail-safe
        default 3; consumers tune it in their cluster YAML, tuning data =
        the EVICTION log lines below) must accumulate first, so one dropped
        packet or a briefly-overloaded server doesn't empty the pool.
        Eviction does NOT scancel the SLURM job — evicted entries move to
        the recovery list, where _reprobe_evicted re-adds them if the
        server comes back.
        """
        threshold = config.get("eviction_failure_threshold", 3)
        with self._pool_cond:
            entries = list(self._pool.get(model, []))

        evicted_any = False
        for entry in entries:
            healthy, err = self._health_check(entry["endpoint"], config)
            if healthy:
                entry["consecutive_health_failures"] = 0
                continue
            fails = entry.get("consecutive_health_failures", 0) + 1
            entry["consecutive_health_failures"] = fails
            if fails < threshold:
                logger.warning(
                    f"Health check failed for {entry['endpoint']} "
                    f"({model.model_id}): {err} "
                    f"({fails}/{threshold} consecutive failures before eviction)")
                continue
            with self._pool_cond:
                pool = self._pool.get(model, [])
                if entry in pool:
                    pool.remove(entry)
                self._evicted.setdefault(model, []).append(entry)
                remaining = len(pool)
            # Structured eviction-event line — the tuning data stream for
            # eviction_failure_threshold.
            logger.warning(
                f"EVICTION model={model.model_id} "
                f"endpoint={entry['endpoint']} job={entry.get('job_id')} "
                f"consecutive_failures={fails} err={err} "
                f"remaining_pool={remaining} — moved to recovery watch "
                f"(re-added if its server recovers)")
            evicted_any = True

        if evicted_any:
            with self._pool_cond:
                self._pool_cond.notify_all()

    def _reprobe_evicted(
        self, model: LLMModel, config: ClusterConfigDict
    ) -> None:
        """Recovery half of T4: an evicted endpoint whose SLURM job is still
        alive gets re-probed each maintenance pass and re-added on recovery
        (transient network partition, vLLM briefly unresponsive under load).
        Once SLURM reports the job gone, the endpoint is dropped for good.

        A recovered endpoint re-enters the pool as available even if it was
        held at eviction time — the old holder's request already failed, and
        its release_endpoint() on a non-pool entry is a harmless no-op.
        """
        with self._pool_cond:
            entries = list(self._evicted.get(model, []))
        if not entries:
            return

        for entry in entries:
            state = self._get_job_state(str(entry.get("job_id", "")), config)
            if state == self.STATE_QUERY_FAILED:
                continue  # unknown — retry next pass
            if state is None:
                with self._pool_cond:
                    evicted = self._evicted.get(model, [])
                    if entry in evicted:
                        evicted.remove(entry)
                logger.info(
                    f"Evicted endpoint {entry['endpoint']} "
                    f"({model.model_id}): job {entry.get('job_id')} has left "
                    f"the queue — dropped permanently.")
                continue

            healthy, _err = self._health_check(entry["endpoint"], config)
            if not healthy:
                continue
            entry["consecutive_health_failures"] = 0
            entry["is_available"] = True
            with self._pool_cond:
                # Teardown race: don't resurrect after shutdown_model/all.
                if model not in self._pool or self._stop_monitor.is_set():
                    continue
                evicted = self._evicted.get(model, [])
                if entry in evicted:
                    evicted.remove(entry)
                self._pool[model].append(entry)
                pool_size = len(self._pool[model])
                self._pool_cond.notify_all()
            logger.info(
                f"RECOVERY model={model.model_id} "
                f"endpoint={entry['endpoint']} job={entry.get('job_id')} "
                f"re-added to pool (pool size: {pool_size})")

    # ==================== SLURM Helpers ====================

    def _generate_sbatch(
        self, model: LLMModel, config: ClusterConfigDict,
        instance_id: int = 0
    ) -> Path:
        """Generate sbatch script for a vLLM server instance."""
        from .constants import MAX_SLURM_TIME_LIMIT

        log_dir = self._resolve_log_dir(config)
        model_safe_name = model.name.lower()
        instance_suffix = f"_i{instance_id}" if instance_id > 0 else ""

        # Combine explicit node exclusions with GPU-type-based exclusions.
        # gpu_types_excluded is resolved to a node list by querying sinfo
        # (the cluster GRES strings carry the GPU type, but not all GPU types
        # are exposed as SLURM features, so `--constraint` can't express this).
        explicit_excluded = list(config.get("excluded_nodes", []))
        type_excluded_nodes = self._resolve_gpu_type_excludes(
            config["partition"], config.get("gpu_types_excluded", []), config)
        all_excluded = sorted(set(explicit_excluded) | set(type_excluded_nodes))
        exclude_directive = (
            f"#SBATCH --exclude={','.join(all_excluded)}"
            if all_excluded else ""
        )

        # Per-cluster wall cap (config key max_slurm_time_limit);
        # MAX_SLURM_TIME_LIMIT stays as the fail-safe fallback (a conservative
        # 8h). Compared NUMERICALLY — string comparison mis-clamps unpadded
        # specs ('4:00:00' > '08:00:00' lexicographically).
        max_wall = config.get("max_slurm_time_limit") or MAX_SLURM_TIME_LIMIT
        time_limit = config["time_limit"]
        limit_s = _slurm_time_to_seconds(time_limit)
        max_s = _slurm_time_to_seconds(max_wall)
        if limit_s is None:
            logger.warning(
                f"unparseable time_limit '{time_limit}' — passing through to "
                f"sbatch unclamped")
        elif max_s is not None and limit_s > max_s:
            logger.warning(
                f"time_limit '{time_limit}' exceeds cluster max "
                f"'{max_wall}', clamping")
            time_limit = max_wall

        vllm_args = [
            f"--model {model.model_id}",
            "--host 0.0.0.0",
            f"--port {config['port']}",
            f"--gpu-memory-utilization {config['gpu_memory_utilization']}",
            f"--max-model-len {config['max_model_len']}",
        ]
        if config.get("dtype"):
            vllm_args.append(f"--dtype {config['dtype']}")
        # On-the-fly weight quantization (e.g. fp8) so a large model fits on ONE
        # GPU — needed on clusters whose partition QOS caps the GPUs per job,
        # where tensor-parallel across GPUs isn't available.
        if config.get("quantization"):
            vllm_args.append(f"--quantization {config['quantization']}")
        if config["num_gpus"] > 1:
            vllm_args.append(f"--tensor-parallel-size {config['num_gpus']}")
        # Some checkpoints (e.g. InternVL) ship custom modeling code in their HF
        # repo and require vLLM to trust it. Per-model config flag, default off.
        if config.get("trust_remote_code"):
            vllm_args.append("--trust-remote-code")
        # Chat template resolution: YAML override → ModelSpec → None.
        # ModelSpec is the source of truth (a model's chat template is an
        # architectural fact, not a deployment choice). YAML can override
        # for ad-hoc experiments. None lets vLLM use the tokenizer's
        # baked-in chat template, which is correct for modern chat-tuned
        # checkpoints (Llama-3+, Qwen, Mistral chat, etc.).
        chat_template_name = config.get("chat_template") or model.chat_template
        if chat_template_name:
            chat_template_dir = Path(__file__).parent / "chat_templates"
            template_file = chat_template_dir / f"{chat_template_name}.jinja"
            # Raise loudly on a missing template file — a silent fallback to
            # vLLM's default template once hid a 400-on-every-request bug.
            if not template_file.exists():
                raise FileNotFoundError(
                    f"chat_template={chat_template_name!r} for {model.model_id} "
                    f"but {template_file} does not exist. Either add the .jinja "
                    f"file or correct the name on ModelSpec / cluster.chat_template.")
            vllm_args.append(f"--chat-template {template_file.resolve()}")
            logger.info(f"Using chat template: {template_file.resolve()}")

        vllm_cmd = "python -m vllm.entrypoints.openai.api_server \\\n    " + \
                   " \\\n    ".join(vllm_args)

        sbatch_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=vllm_{model_safe_name}{instance_suffix}",
            f"#SBATCH --partition={config['partition']}",
        ]
        if exclude_directive:
            sbatch_lines.append(exclude_directive)
        # Positive feature constraint — e.g. force `a100@80g` when a model
        # needs 80GB cards specifically (Llama-3.3-70B at FP16 = 140GB
        # weights, won't fit on 2× 40GB). SLURM constraint syntax supports
        # AND (`&`), OR (`|`), and feature names directly.
        gpu_constraint = config.get("gpu_constraint")
        if gpu_constraint:
            sbatch_lines.append(f"#SBATCH --constraint={gpu_constraint}")
        # Some clusters gate multi-GPU serving behind a specific QOS (the default
        # partition QOS capping GPUs per job). Config-driven.
        qos = config.get("qos")
        if qos:
            sbatch_lines.append(f"#SBATCH --qos={qos}")
        # GPU request directive — some clusters use `--gres=gpu:N`; others with
        # homogeneous partitions want `--gpus=N` (partition selects the GPU
        # type). Config-driven.
        if config.get("gpu_request_style", "gres") == "gpus":
            gpu_directive = f"#SBATCH --gpus={config['num_gpus']}"
        else:
            gpu_directive = f"#SBATCH --gres=gpu:{config['num_gpus']}"

        # Env setup — default is a conda-module setup (anaconda3 module +
        # `source activate <env>`). A cluster profile can REPLACE both lines via
        # config['env_setup'] (e.g. a cuda module + a uv-venv activate).
        env_setup = config.get("env_setup")
        if env_setup:
            env_lines = list(env_setup)
        else:
            env_lines = [
                f"module load anaconda3/2024.06 {config['cuda_module']}",
                f"source activate {config['conda_env']}",
            ]

        # HF cache + offline mode. Clusters whose compute nodes have no internet
        # force offline; online compute nodes set hf_offline=false to pull live.
        hf_lines = [f"export HF_HOME=\"${{HF_HOME:-{config['hf_home']}}}\""]
        # Gated-repo auth (wildguard, llama_guard_3, …): the HF libraries read
        # HF_TOKEN, but the project's injected secret is named HUGGINGFACE_TOKEN
        # (the orchestrator's env injection sets it; sbatch's default env export
        # carries it into this server job). Alias it so live weight pulls
        # authenticate; harmless in offline mode. Without this, gated models 401
        # when hf_offline=false pulls weights (observed 2026-07-18).
        hf_lines.append('export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"')
        if config.get("hf_offline", True):
            hf_lines += ["export HF_HUB_OFFLINE=1", "export TRANSFORMERS_OFFLINE=1"]

        sbatch_lines.extend([
            "#SBATCH --nodes=1",
            gpu_directive,
            f"#SBATCH --cpus-per-task={config['cpus_per_task']}",
            f"#SBATCH --mem={config['mem_gb']}GB",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output={log_dir}/vllm_{model_safe_name}{instance_suffix}_%j.out",
            f"#SBATCH --error={log_dir}/vllm_{model_safe_name}{instance_suffix}_%j.err",
            "",
            "# Setup environment",
            f"mkdir -p {log_dir}",
            *env_lines,
            "",
            "# HuggingFace cache (+ offline mode where compute nodes have no internet)",
            *hf_lines,
            "",
            "# Start vLLM server (blocks until killed by scancel or time limit)",
            vllm_cmd,
        ])

        script = "\n".join(sbatch_lines) + "\n"

        # slurmstepd opens the --output/--error files (relative to the submit
        # cwd) BEFORE the script body runs, so the in-script `mkdir -p` is
        # too late on a fresh directory — create it at generation time.
        log_dir.mkdir(parents=True, exist_ok=True)

        sbatch_path = self._sbatch_dir / f"vllm_{model_safe_name}{instance_suffix}.sbatch"
        sbatch_path.write_text(script)
        logger.debug(f"Generated sbatch at {sbatch_path}")
        return sbatch_path

    def _submit_sbatch(self, sbatch_path: Path) -> str:
        """Submit sbatch script and return the SLURM job ID.

        The login-node SLURM controller is intermittently slow or flaky
        (observed >30s answers and transient "Socket timed out on send/recv
        operation" errors under load), which used to kill the whole run on a
        single blip. Retry BOTH timeouts and nonzero exits a few times with
        a short backoff — transient controller errors are indistinguishable
        from permanent ones cheaply, and a genuinely bad script only wastes
        ~12s of retries.
        """
        last_err: Optional[str] = None
        for attempt in range(4):
            if attempt:
                time.sleep(2 * attempt)  # 2s, 4s, 6s between attempts
            try:
                result = subprocess.run(
                    ["sbatch", str(sbatch_path)],
                    capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                last_err = "sbatch timed out after 120s"
                logger.warning(
                    f"sbatch slow (attempt {attempt + 1}/4) for "
                    f"{sbatch_path.name}; retrying")
                continue
            except FileNotFoundError:
                raise RuntimeError(
                    "sbatch command not found. "
                    "Are you running this on the cluster login node?")
            if result.returncode == 0:
                return result.stdout.strip().split()[-1]
            last_err = result.stderr.strip() or f"exit code {result.returncode}"
            logger.warning(
                f"sbatch failed (attempt {attempt + 1}/4) for "
                f"{sbatch_path.name}: {last_err}; retrying")
        raise RuntimeError(f"sbatch failed after 4 attempts: {last_err}")

    def _scancel(self, job_id: str) -> None:
        """Cancel one SLURM job; failures are logged, never raised."""
        try:
            subprocess.run(
                ["scancel", str(job_id)], capture_output=True, timeout=10)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning(f"Could not cancel job {job_id}")

    def _resolve_job_node(
        self, job_id: str, config: ClusterConfigDict
    ) -> Optional[str]:
        """
        Get the IP address or hostname of the node running a SLURM job.

        Resolves via scontrol: job → NodeList → NodeAddr (IP).
        Falls back to hostname if IP resolution fails.
        """
        cmd_timeout = config.get("slurm_cmd_timeout", 15)

        try:
            result = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True, text=True, timeout=cmd_timeout)
            if result.returncode != 0:
                return None

            # `\b` so "NodeList=" doesn't match as a suffix of "ReqNodeList="
            # or "ExcNodeList=" (which scontrol also emits on every job, e.g.
            # "ReqNodeList=(null) ExcNodeList=node[01-04] NodeList=node07").
            # Without the boundary, re.search hits ReqNodeList first and
            # captures "(null)", short-circuiting node resolution forever.
            match = re.search(r"\bNodeList=(\S+)", result.stdout)
            if not match:
                return None
            node = match.group(1)
            if not node or node == "(null)":
                return None

            # Resolve node name → IP address
            result = subprocess.run(
                ["scontrol", "show", "node", node],
                capture_output=True, text=True, timeout=cmd_timeout)
            if result.returncode != 0:
                return node

            addr_match = re.search(r"\bNodeAddr=(\S+)", result.stdout)
            if addr_match:
                addr = addr_match.group(1)
                if addr and addr != node:
                    logger.info(f"Resolved node {node} -> IP {addr}")
                    return addr

            return node

        except subprocess.TimeoutExpired:
            logger.warning(f"scontrol timed out for job {job_id}")
            return None
        except FileNotFoundError:
            logger.warning("scontrol command not found")
            return None

    def _resolve_gpu_type_excludes(
        self, partition: str, excluded_types: List[str],
        config: ClusterConfigDict
    ) -> List[str]:
        """
        Resolve `gpu_types_excluded` to a concrete node-name list by querying sinfo.

        The cluster encodes GPU type in the GRES string (`gpu:<type>:N`) but not
        always as a SLURM feature, so SBATCH --constraint can't filter by type.
        Instead we expand each excluded GPU type to the set of nodes carrying it
        and merge into --exclude=.

        Result is cached per (partition, frozenset(excluded_types)).
        """
        if not excluded_types:
            return []

        cache_key = (partition, frozenset(excluded_types))
        if cache_key in self._gpu_type_exclude_cache:
            return self._gpu_type_exclude_cache[cache_key]

        cmd_timeout = config.get("slurm_cmd_timeout", 15)

        try:
            result = subprocess.run(
                ["sinfo", "--partition", partition,
                 "--noheader", "-o", "%n %G"],
                capture_output=True, text=True, timeout=cmd_timeout)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(
                f"sinfo failed while resolving gpu_types_excluded: {e}. "
                f"GPU-type filter will be a no-op for this run.")
            return []

        if result.returncode != 0:
            logger.warning(
                f"sinfo returncode={result.returncode} resolving "
                f"gpu_types_excluded: {result.stderr.strip()}")
            return []

        excluded_set = set(excluded_types)
        nodes: List[str] = []
        seen: set = set()
        for line in result.stdout.strip().splitlines():
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            node, gres = parts[0], parts[1]
            # GRES form: "gpu:v100-pcie:2(S:0-1)" or "gpu:a100:4" or "(null)"
            m = re.match(r"gpu:([^:()]+):", gres)
            if not m:
                continue
            gpu_type = m.group(1)
            if gpu_type in excluded_set and node not in seen:
                nodes.append(node)
                seen.add(node)

        logger.info(
            f"gpu_types_excluded={excluded_types} resolved to "
            f"{len(nodes)} node(s) on partition '{partition}'")
        self._gpu_type_exclude_cache[cache_key] = nodes
        return nodes

    # Sentinel: the squeue QUERY itself failed (controller slow / timeout) —
    # distinct from None, which means the job has left the queue (finished or
    # failed). Callers must treat this as "unknown, retry later", never as a
    # dead job — a live server must not be blackholed by one slow squeue.
    STATE_QUERY_FAILED = "QUERY_FAILED"

    def _get_job_state(
        self, job_id: str, config: ClusterConfigDict
    ) -> Optional[str]:
        """Get the current SLURM state of a job (RUNNING, PENDING, etc.).
        None = the job is no longer in the queue; STATE_QUERY_FAILED = squeue
        itself failed (transient — retry)."""
        cmd_timeout = config.get("slurm_cmd_timeout", 15)

        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "--noheader", "--format=%T"],
                capture_output=True, text=True, timeout=cmd_timeout)
            state = result.stdout.strip()
            return state or None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return self.STATE_QUERY_FAILED

    def _health_check(
        self, endpoint: str, config: ClusterConfigDict
    ) -> tuple[bool, Optional[str]]:
        """
        Check if the vLLM server is responding at the /health endpoint.

        Returns:
            (is_healthy, error_message_or_None)
        """
        import urllib.request
        import urllib.error

        check_timeout = config.get("health_check_timeout", 5)

        base_url = endpoint.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        health_url = f"{base_url}/health"
        try:
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            req = urllib.request.Request(health_url, method="GET")
            with opener.open(req, timeout=check_timeout) as resp:
                return (resp.status == 200, None)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            return (False, str(e))
