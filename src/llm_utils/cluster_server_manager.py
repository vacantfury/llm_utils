"""
Manages vLLM server lifecycle on SLURM cluster.

Handles:
- Auto-generating sbatch scripts for vLLM servers
- Submitting SLURM jobs and tracking their IDs
- Discovering endpoints via scontrol (no shared files needed)
- Multi-instance support: N servers per model on different ports
- Dynamic server pool with acquire/release endpoint allocation
- Background monitoring: servers are added to the pool as they become healthy
"""
import re
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


class ClusterModelServerManager:
    """
    Manages vLLM server lifecycle on SLURM cluster with dynamic endpoint pool.

    Supports multiple server instances per model for parallel task execution.
    Each instance gets its own SLURM job, GPU, and port.

    Servers are added to the pool dynamically as they become healthy.
    Tasks acquire/release endpoints — no blocking on all servers at startup.

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
        self._pool_lock = threading.Lock()
        self._pool_changed = threading.Event()

        self.model_configs: Dict[LLMModel, ClusterConfigDict] = {}

        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()

        self._sbatch_dir = Path(tempfile.mkdtemp(prefix="vllm_sbatch_"))

        # Cache: (partition, frozenset(excluded_types)) -> [node names with excluded GPU]
        self._gpu_type_exclude_cache: Dict[tuple, List[str]] = {}

    # ==================== Public API ====================

    def start_server(self, model: LLMModel, config: ClusterConfigDict) -> None:
        """
        Start vLLM server(s) for the given model.

        Submits SLURM jobs and starts a background monitor thread.
        Returns immediately — does NOT wait for servers to be healthy.

        Args:
            model: The cluster model to serve.
            config: Cluster config dict (from load_conf("llm", section="cluster")).
        """
        if model in self._jobs and self._jobs[model]:
            logger.info(f"Servers already submitted for {model.model_id}")
            return

        if model.provider != Provider.NU_CLUSTER:
            raise ValueError(
                f"{model.model_id} is not a cluster model (provider: {model.provider})")

        self.model_configs[model] = config

        num_instances = config["num_instances"]
        base_port = config["port"]

        logger.info(
            f"Starting {num_instances} vLLM server(s) for {model.model_id} "
            f"(ports {base_port}-{base_port + num_instances - 1})")

        self._jobs[model] = []
        self._pool[model] = []

        for i in range(num_instances):
            instance_port = base_port + i
            instance_config = {**config, "port": instance_port}

            sbatch_path = self._generate_sbatch(model, instance_config, instance_id=i)
            job_id = self._submit_sbatch(sbatch_path)

            self._jobs[model].append({
                "job_id": job_id,
                "port": instance_port,
                "instance_id": i,
                "discovered": False,
            })
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

        while time.time() < deadline:
            with self._pool_lock:
                for entry in self._pool.get(model, []):
                    if entry["is_available"]:
                        entry["is_available"] = False
                        logger.debug(f"Acquired endpoint {entry['endpoint']}")
                        return entry["endpoint"]

            # Fail fast if NO server was ever submitted for this model — a
            # serve-discovery miss (e.g. a cluster judge selected via judge_model
            # that the orchestrator never added to its cluster-model set). Without
            # this, acquire blocks the full cluster_server_endpoint_timeout (~2.8h)
            # on a pool that can never fill (2026-07-16 wildguard-judge hang).
            jobs = self._jobs.get(model, [])
            if not jobs:
                raise RuntimeError(
                    f"No vLLM server was ever started for {model.model_id}: a task "
                    f"requested it (e.g. as a judge/guard) but the orchestrator did "
                    f"not add it to its cluster-model set (serve-discovery miss). "
                    f"Check judge_model/guard_model serve wiring in "
                    f"_required_cluster_models_for_task.")
            # Short-circuit on all-failed: if every submitted job for this
            # model is already discovered (resolved by monitor) and the pool
            # is still empty, every one of them failed — no point waiting.
            # Mirrors wait_for_first_server's check; lets tasks fail fast
            # when their judge can't start (e.g. QoS-blocked) instead of
            # hanging up to cluster_server_endpoint_timeout (10000s default).
            if all(j["discovered"] for j in jobs):
                with self._pool_lock:
                    pool_empty = not self._pool.get(model)
                if pool_empty:
                    raise RuntimeError(
                        f"All {len(jobs)} server job(s) for "
                        f"{model.model_id} failed during discovery. "
                        f"Check logs/vllm_{model.name.lower()}_*.err for "
                        f"startup errors (config mismatch, QoS-blocked, OOM, etc.).")

            remaining = deadline - time.time()
            wait_time = min(wait_interval, max(remaining, 0))
            if wait_time <= 0:
                break
            self._pool_changed.wait(timeout=wait_time)
            self._pool_changed.clear()

        raise RuntimeError(
            f"No endpoint available for {model.model_id} within {timeout}s. "
            f"Pool has {len(self._pool.get(model, []))} server(s), "
            f"all busy or none started.")

    def release_endpoint(self, model: LLMModel, endpoint: str) -> None:
        """Release an endpoint back to the pool, marking it as available."""
        with self._pool_lock:
            for entry in self._pool.get(model, []):
                if entry["endpoint"] == endpoint:
                    entry["is_available"] = True
                    logger.debug(f"Released endpoint {endpoint}")
                    break
        self._pool_changed.set()

    def get_num_instances(self, model: LLMModel) -> int:
        """Get the number of submitted server instances for a model."""
        return len(self._jobs.get(model, []))

    def get_num_ready(self, model: LLMModel) -> int:
        """Get the number of healthy, pool-registered endpoints for a model."""
        with self._pool_lock:
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

        while time.time() < deadline:
            with self._pool_lock:
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
                    f"(serve-discovery miss). Check judge_model/guard_model "
                    f"serve wiring in _required_cluster_models_for_task.")
            # Short-circuit on all-failed: if every submitted job for this
            # model is already discovered (resolved by monitor) and the pool
            # is still empty, every one of them failed — no point waiting.
            if all(j["discovered"] for j in jobs):
                with self._pool_lock:
                    pool_empty = not self._pool.get(model)
                if pool_empty:
                    raise RuntimeError(
                        f"All {len(jobs)} server job(s) for "
                        f"{model.model_id} failed during discovery. "
                        f"Check logs/vllm_{model.name.lower()}_*.err for "
                        f"startup errors (config mismatch, OOM, etc.).")

            # Periodic progress log so an extended wait isn't silent (the
            # monitor only logs successes/failures; PENDING + slow-server
            # would otherwise produce no orchestrator-side output for many
            # minutes).
            now = time.time()
            if now - last_progress_log >= progress_interval:
                with self._pool_lock:
                    pool_size = len(self._pool.get(model, []))
                jobs_now = self._jobs.get(model, [])
                discovered = sum(1 for j in jobs_now if j["discovered"])
                elapsed = int(now - start)
                logger.info(
                    f"Still waiting for {model.model_id}: elapsed={elapsed}s, "
                    f"jobs_submitted={len(jobs_now)}, jobs_discovered="
                    f"{discovered}, pool_size={pool_size}. "
                    f"(See per-job monitor lines above for what's blocking.)")
                last_progress_log = now

            remaining = deadline - time.time()
            if remaining <= 0:
                break
            self._pool_changed.wait(timeout=min(10, remaining))
            self._pool_changed.clear()

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
        }

        if model not in self._jobs:
            return status

        jobs = self._jobs[model]
        status["job_ids"] = [j["job_id"] for j in jobs]
        status["num_submitted"] = len(jobs)

        with self._pool_lock:
            pool = self._pool.get(model, [])
            status["num_ready"] = len(pool)
            status["num_available"] = sum(1 for e in pool if e["is_available"])
            status["endpoints"] = [e["endpoint"] for e in pool]

        if status["num_ready"] == 0:
            status["state"] = "pending"
        elif status["num_ready"] < status["num_submitted"]:
            status["state"] = "partially_ready"
        else:
            status["state"] = "ready"

        return status

    def shutdown_all(self) -> None:
        """Cancel all active SLURM jobs and clean up."""
        self._stop_monitor.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

        for model, jobs in self._jobs.items():
            for job_info in jobs:
                job_id = job_info["job_id"]
                logger.info(f"Cancelling SLURM job {job_id} ({model.model_id})")
                try:
                    subprocess.run(
                        ["scancel", job_id],
                        capture_output=True, timeout=10)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    logger.warning(f"Could not cancel job {job_id}")

        self._jobs.clear()
        with self._pool_lock:
            self._pool.clear()
        self._pool_changed.set()
        logger.info("All vLLM servers shut down")

    def shutdown_model(self, model: LLMModel) -> None:
        """Cancel all SLURM jobs for a single model and clear its pool entries.

        Mid-run teardown: lets the orchestrator free a heavy model's GPU
        allocation as soon as the last task needing it completes, instead of
        holding 4× A100s pinned for the entire experiment lifetime.

        Safe to call on a model that's already been torn down (no-op).
        """
        jobs = self._jobs.pop(model, [])
        for job_info in jobs:
            job_id = job_info["job_id"]
            logger.info(
                f"Cancelling SLURM job {job_id} ({model.model_id}) [mid-run]")
            try:
                subprocess.run(
                    ["scancel", job_id], capture_output=True, timeout=10)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning(f"Could not cancel job {job_id}")
        with self._pool_lock:
            self._pool.pop(model, None)
        self._pool_changed.set()

    def __del__(self):
        if self._jobs:
            self.shutdown_all()

    # ==================== Background Monitor ====================

    def _monitor_loop(self) -> None:
        """Background thread: discover new servers AND keep checking pool health.

        Phase 1 (discovery): poll SLURM jobs and add healthy endpoints to the
        pool as they become reachable. Each job is "discovered" exactly once.

        Phase 2 (maintenance): periodically re-ping every pool entry's /health.
        If a previously-discovered server stops responding (vLLM OOM'd, GPU
        ECC'd, etc.), evict it from the pool. Without this, acquire_endpoint()
        would hand out a stale dead endpoint and the task would hang on the
        first request rather than waiting for a fresh one.

        Runs until _stop_monitor is set (i.e. until shutdown_all()).
        """
        config = next(iter(self.model_configs.values()), {})
        poll_interval = config.get("monitor_poll_interval", 10)
        # Cadence at which discovered endpoints get re-health-checked.
        # 60s default keeps overhead near zero for typical pool sizes.
        recheck_interval = config.get("health_recheck_interval", 60)
        last_recheck = 0.0

        while not self._stop_monitor.is_set():
            # ---- Phase 1: discovery ----
            for model, jobs in list(self._jobs.items()):
                for job_info in jobs:
                    if job_info["discovered"]:
                        continue

                    job_id = job_info["job_id"]
                    port = job_info["port"]
                    instance_id = job_info["instance_id"]
                    instance_suffix = f"[{instance_id}]" if instance_id > 0 else ""

                    state = self._get_job_state(job_id)

                    if state is None:
                        model_safe_name = model.name.lower()
                        logger.warning(
                            f"Server{instance_suffix} (job {job_id}) failed. "
                            f"Check logs/vllm_{model_safe_name}_*.err")
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

                    node = self._resolve_job_node(job_id)
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
                    healthy, err = self._health_check(endpoint)

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
                        with self._pool_lock:
                            self._pool[model].append({
                                "endpoint": endpoint,
                                "is_available": True,
                                "job_id": job_id,
                            })
                        job_info["discovered"] = True
                        logger.info(
                            f"Server{instance_suffix} ready at {endpoint} "
                            f"(pool size: {len(self._pool[model])})")
                        self._pool_changed.set()

            # ---- Phase 2: re-health-check discovered endpoints ----
            now = time.time()
            if now - last_recheck >= recheck_interval:
                last_recheck = now
                self._recheck_pool_health()

            self._stop_monitor.wait(timeout=poll_interval)

    def _recheck_pool_health(self) -> None:
        """Re-ping every pool entry's /health; evict any that fail.

        Catches mid-run vLLM crashes (OOM, GPU error, network blip) so that
        acquire_endpoint() doesn't hand out a dead URL. Eviction does NOT
        scancel the SLURM job — the job is presumed already dying or done,
        and a future shutdown_all/_model will clean it up.

        Single-failure eviction is intentional: /health is a constant-time
        endpoint, so a 5s timeout failing means the server really is gone.
        """
        with self._pool_lock:
            snapshot = {m: list(entries) for m, entries in self._pool.items()}

        evicted_any = False
        for model, entries in snapshot.items():
            for entry in entries:
                healthy, err = self._health_check(entry["endpoint"])
                if healthy:
                    continue
                with self._pool_lock:
                    pool = self._pool.get(model, [])
                    self._pool[model] = [
                        e for e in pool if e["endpoint"] != entry["endpoint"]
                    ]
                logger.warning(
                    f"Mid-run health check failed for {entry['endpoint']} "
                    f"({model.model_id}): {err}. Evicted from pool "
                    f"(remaining: {len(self._pool.get(model, []))}).")
                evicted_any = True

        if evicted_any:
            self._pool_changed.set()

    # ==================== SLURM Helpers ====================

    def _generate_sbatch(
        self, model: LLMModel, config: ClusterConfigDict,
        instance_id: int = 0
    ) -> Path:
        """Generate sbatch script for a vLLM server instance."""
        from .constants import MAX_SLURM_TIME_LIMIT

        model_safe_name = model.name.lower()
        instance_suffix = f"_i{instance_id}" if instance_id > 0 else ""

        # Combine explicit node exclusions with GPU-type-based exclusions.
        # gpu_types_excluded is resolved to a node list by querying sinfo
        # (the cluster GRES strings carry the GPU type, but not all GPU types
        # are exposed as SLURM features, so `--constraint` can't express this).
        explicit_excluded = list(config.get("excluded_nodes", []))
        type_excluded_nodes = self._resolve_gpu_type_excludes(
            config["partition"], config.get("gpu_types_excluded", []))
        all_excluded = sorted(set(explicit_excluded) | set(type_excluded_nodes))
        exclude_directive = (
            f"#SBATCH --exclude={','.join(all_excluded)}"
            if all_excluded else ""
        )

        # Per-cluster wall cap (conf/clusters/<profile>.yaml::max_slurm_time_limit);
        # MAX_SLURM_TIME_LIMIT stays as the fail-safe fallback (NURC's 8h).
        max_wall = config.get("max_slurm_time_limit") or MAX_SLURM_TIME_LIMIT
        time_limit = config["time_limit"]
        if time_limit and time_limit > max_wall:
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
        # GPU — required here because the `gpu` PartitionQOS caps 1 GPU/job and
        # this account can't request tensor-parallel across GPUs.
        if config.get("quantization"):
            vllm_args.append(f"--quantization {config['quantization']}")
        if config["num_gpus"] > 1:
            vllm_args.append(f"--tensor-parallel-size {config['num_gpus']}")
        # Some checkpoints (e.g. InternVL) ship custom modeling code in their HF
        # repo and require vLLM to trust it. Per-model flag in conf/llm, default off.
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
            # Existence is also validated at config-load time by
            # LLMConfig._check_chat_template_exists; if we get here with a
            # missing file, that validator was bypassed (e.g., ad-hoc
            # callers). Raise loudly — silent fallback to vLLM's default
            # is how the HarmBench 400-on-every-request bug stayed hidden.
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
        # The `gpu` PartitionQOS caps 1 GPU/job; multi-GPU serving must run under
        # the `multigpu` QOS (no per-job GPU cap, up to 8 GPUs across 4 jobs).
        qos = config.get("qos")
        if qos:
            sbatch_lines.append(f"#SBATCH --qos={qos}")
        # GPU request directive — NURC uses `--gres=gpu:N`; AICR's homogeneous
        # partitions want `--gpus=N` (partition selects the GPU type). Config-driven.
        if config.get("gpu_request_style", "gres") == "gpus":
            gpu_directive = f"#SBATCH --gpus={config['num_gpus']}"
        else:
            gpu_directive = f"#SBATCH --gres=gpu:{config['num_gpus']}"

        # Env setup — default is NURC's conda (anaconda3 module + `source activate <env>`).
        # A cluster profile can REPLACE both lines via config['env_setup'] (AICR: cuda
        # module + `source .venv/bin/activate` for the uv venv).
        env_setup = config.get("env_setup")
        if env_setup:
            env_lines = list(env_setup)
        else:
            env_lines = [
                f"module load anaconda3/2024.06 {config['cuda_module']}",
                f"source activate {config['conda_env']}",
            ]

        # HF cache + offline mode. NURC compute nodes have no internet (offline forced);
        # a cluster with online compute nodes sets hf_offline=false to pull weights live.
        hf_lines = [f"export HF_HOME=\"${{HF_HOME:-{config['hf_home']}}}\""]
        # Gated-repo auth (wildguard, llama_guard_3, …): the HF libraries read
        # HF_TOKEN, but the project's injected secret is named HUGGINGFACE_TOKEN
        # (op run sets it on the orchestrator; sbatch's default env export carries
        # it into this server job). Alias it so live weight pulls authenticate;
        # harmless in offline mode. Without this, gated models 401 when
        # hf_offline=false pulls weights (observed on AICR 2026-07-18).
        hf_lines.append('export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"')
        if config.get("hf_offline", True):
            hf_lines += ["export HF_HUB_OFFLINE=1", "export TRANSFORMERS_OFFLINE=1"]

        sbatch_lines.extend([
            "#SBATCH --nodes=1",
            gpu_directive,
            f"#SBATCH --cpus-per-task={config['cpus_per_task']}",
            f"#SBATCH --mem={config['mem_gb']}GB",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output=logs/vllm_{model_safe_name}{instance_suffix}_%j.out",
            f"#SBATCH --error=logs/vllm_{model_safe_name}{instance_suffix}_%j.err",
            "",
            "# Setup environment",
            "mkdir -p logs",
            *env_lines,
            "",
            "# HuggingFace cache (+ offline mode where compute nodes have no internet)",
            *hf_lines,
            "",
            "# Start vLLM server (blocks until killed by scancel or time limit)",
            vllm_cmd,
        ])

        script = "\n".join(sbatch_lines) + "\n"

        sbatch_path = self._sbatch_dir / f"vllm_{model_safe_name}{instance_suffix}.sbatch"
        sbatch_path.write_text(script)
        logger.debug(f"Generated sbatch at {sbatch_path}")
        return sbatch_path

    def _submit_sbatch(self, sbatch_path: Path) -> str:
        """Submit sbatch script and return the SLURM job ID.

        The login-node SLURM controller is intermittently slow to answer
        `sbatch` (observed >30s under load), which used to kill the whole run
        on a single blip. Retry a few times with a generous per-attempt
        timeout instead of failing hard.
        """
        last_err = None
        for attempt in range(4):
            try:
                result = subprocess.run(
                    ["sbatch", str(sbatch_path)],
                    capture_output=True, text=True, timeout=120)
                if result.returncode != 0:
                    raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
                return result.stdout.strip().split()[-1]
            except subprocess.TimeoutExpired as e:
                last_err = e
                logger.warning(
                    f"sbatch slow (attempt {attempt + 1}/4) for "
                    f"{sbatch_path.name}; retrying")
                continue
            except FileNotFoundError:
                raise RuntimeError(
                    "sbatch command not found. "
                    "Are you running this on the cluster login node?")
        raise RuntimeError(f"sbatch timed out after 4 attempts: {last_err}")

    def _resolve_job_node(self, job_id: str) -> Optional[str]:
        """
        Get the IP address or hostname of the node running a SLURM job.

        Resolves via scontrol: job → NodeList → NodeAddr (IP).
        Falls back to hostname if IP resolution fails.
        """
        config = next(iter(self.model_configs.values()), {})
        cmd_timeout = config.get("slurm_cmd_timeout", 15)

        try:
            result = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True, text=True, timeout=cmd_timeout)
            if result.returncode != 0:
                return None

            # `\b` so "NodeList=" doesn't match as a suffix of "ReqNodeList="
            # or "ExcNodeList=" (which scontrol also emits on every job, e.g.
            # "ReqNodeList=(null) ExcNodeList=c[2204-2207],d[...] NodeList=d4055").
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
        self, partition: str, excluded_types: List[str]
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

        config = next(iter(self.model_configs.values()), {})
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

    def _get_job_state(self, job_id: str) -> Optional[str]:
        """Get the current SLURM state of a job (RUNNING, PENDING, etc.)."""
        config = next(iter(self.model_configs.values()), {})
        cmd_timeout = config.get("slurm_cmd_timeout", 15)

        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "--noheader", "--format=%T"],
                capture_output=True, text=True, timeout=cmd_timeout)
            state = result.stdout.strip()
            return state or None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _health_check(self, endpoint: str) -> tuple[bool, Optional[str]]:
        """
        Check if the vLLM server is responding at the /health endpoint.

        Returns:
            (is_healthy, error_message_or_None)
        """
        import urllib.request
        import urllib.error

        config = next(iter(self.model_configs.values()), {})
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
