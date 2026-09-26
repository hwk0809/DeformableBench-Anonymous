"""Out-of-process policy-server launchers (lingbot / gr00t / fastwam / lerobot_remote).

Moved verbatim out of ``bench/eval_sim.py`` so the real-robot entry point
(``bench/real_eval.py``) can reuse the exact same launch logic without importing
the sim stack (eval_sim pulls in collect_data -> newton/Kit, which the robot box
has no reason to load). ``eval_sim`` re-imports these under their original names,
so its own code is unchanged.

Each ``_launch_*`` takes the eval's parsed args namespace plus a loguru-style
logger, spawns the server in its own conda env with ``start_new_session=True``
(so the caller can kill the whole process group), blocks until the port accepts
TCP, and returns the ``Popen`` — or ``None`` when no ``--*_server_python`` was
given, meaning "connect to an already-running server".
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional


def _wait_tcp(host: str, port: int, timeout_s: float, poll_s: float = 2.0, proc=None) -> None:
    """Block until the port accepts TCP -- and give up the moment ``proc`` dies.

    Without the ``proc`` check this loop cannot fail for the one reason that
    matters: if the server we just spawned exits (EADDRINUSE, bad checkpoint,
    OOM), *some other* listener may still answer on that port and the loop
    reports success.
    """
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"server exited with code {proc.returncode} before {host}:{port} came up; "
                f"its stderr is above (a bind error means the port was already taken)")
        try:
            with socket.create_connection((host, int(port)), timeout=5):
                return
        except OSError as e:
            last = e
            time.sleep(poll_s)
    raise TimeoutError(f"TCP service not ready at {host}:{port} after {timeout_s}s: {last}")


def _port_is_free(port: int) -> bool:
    """Could a server still bind this port?

    Probing by bind is the authority here because it is the very call the server
    is about to make: ``SO_REUSEADDR`` does not let a second process listen on an
    address another process already holds, so a successful probe and a successful
    server start succeed or fail together.
    """
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("0.0.0.0", int(port)))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _port_holder(port: int) -> tuple[Optional[int], list[str]]:
    """Best-effort ``(pid, argv)`` of the listener on ``port``.

    ``argv`` comes straight off ``/proc/<pid>/cmdline``, which is NUL-separated --
    keeping it as a list is what makes it comparable to the command we are about
    to spawn. Flattening it to a string and re-splitting would not round-trip
    (``shlex`` eats the quotes inside a ``-c`` payload). Either half may come back
    empty, since ``ss`` hides the pid of a socket owned by another user.
    """
    try:
        out = subprocess.run(["ss", "-lptnH", f"sport = :{int(port)}"],
                             capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"pid=(\d+)", out)
        if not m:
            return None, []
        pid = int(m.group(1))
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace").split("\0")
        while argv and argv[-1] == "":
            argv.pop()
        return pid, argv
    except Exception:
        return None, []


def _show(argv: list[str]) -> str:
    return shlex.join(argv) if argv else "unidentified (owned by another user?)"


def _sid(pid: int) -> Optional[int]:
    try:
        return os.getsid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None


def _is_descendant(pid: int, ancestor: int) -> bool:
    """Return whether pid is in ancestor's current process tree."""
    seen: set[int] = set()
    current = int(pid)
    while current > 1 and current not in seen:
        if current == int(ancestor):
            return True
        seen.add(current)
        try:
            stat = Path(f"/proc/{current}/stat").read_text()
            # The comm field may contain spaces/parentheses; split after its final ')'.
            fields = stat.rsplit(")", 1)[1].split()
            current = int(fields[1])  # state is fields[0], ppid is fields[1]
        except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
            return False
    return current == int(ancestor)


def _die_with_parent():
    """``preexec_fn`` asking the kernel to SIGTERM this child when its parent dies.

    The only mechanism here that survives the parent being SIGKILLed (terminal
    killed, ssh dropped, OOM killer) -- a Python ``finally`` never runs in those
    cases, and the server, sitting in its own session, is not in the process group
    the terminal signals. It also needs no cooperation from the server itself,
    which matters because two of the four are third-party entry points.

    Runs in the child after ``setsid``; PDEATHSIG survives both that and the exec.
    SIGTERM rather than SIGKILL so lingbot's torchrun gets to forward it to its
    workers, which do not inherit PDEATHSIG themselves. The one case this leaves
    open -- a server that ignores SIGTERM *and* a parent that is SIGKILLed -- does
    not arise today: none of the four servers installs a SIGTERM handler.
    """
    parent = os.getpid()  # captured pre-fork: in the child this is the parent's pid
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return None  # non-Linux or no libc: fall back to the finally-based teardown

    # Everything the child touches is resolved here, before the fork. Importing or
    # loading a library between fork and exec can deadlock on a lock another thread
    # held at fork time, and this runs inside real_eval, which is multi-process.
    def _fn():
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
        # The parent may have died in the fork->prctl window, in which case the
        # signal already came and went. Check by hand rather than hang around.
        if os.getppid() != parent:
            os._exit(1)

    return _fn


def _attach(name: str, port: int, logger):
    """No ``--*_server_python`` given: use whatever already listens, but say what it is.

    Silently inheriting an unnamed server is how a run ends up labelled with one
    checkpoint and executed with another, so the identity goes in the log even on
    the path where we deliberately do not manage the process.
    """
    pid, argv = _port_holder(port)
    if pid is None and _port_is_free(port):
        logger.warning(f"[eval] {name}: nothing is listening on :{port} yet "
                       f"(no --{name}_server_python given, so none will be launched)")
    else:
        logger.info(f"[eval] {name}: attaching to the server already on :{port} "
                    f"(pid={pid}) -> {_show(argv)}")
    return None


def _spawn_server(name: str, cmd, *, host: str, port, wait_s: float, logger,
                  cwd=None, env=None):
    """Start a policy server and prove the port really ended up belonging to it.

    The failure this exists to prevent: a server left over from an earlier eval
    still owns the port, the new one dies on EADDRINUSE, and a readiness check
    that only asks "does the port accept TCP?" answers yes -- so the client
    spends the whole run talking to the *previous* checkpoint and every number it
    produces is mislabelled. That is not hypothetical: a fastwam server survived
    seven days on :5570 and answered for three different ``--run_dir``s.
    """
    cmd = [str(c) for c in cmd]
    port = int(port)
    wait_host = host if host != "0.0.0.0" else "127.0.0.1"

    if not _port_is_free(port):
        pid, argv = _port_holder(port)
        if argv == cmd:
            logger.warning(
                f"[eval] {name}: :{port} already runs this exact command (pid={pid}); reusing it. "
                f"Weights are not re-read, so restart it if the checkpoint changed on disk.")
            return None
        raise RuntimeError(
            f"refusing to launch the {name} server: :{port} is already held by pid={pid or '?'}.\n"
            f"  holder: {_show(argv)}\n"
            f"  wanted: {shlex.join(cmd)}\n"
            f"Kill the stale server (kill {pid}) or pick another port. Continuing would "
            f"evaluate the OTHER checkpoint under this run's name.")

    logger.info(f"[eval] launching {name} server on :{port}: {shlex.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, start_new_session=True,
                            preexec_fn=_die_with_parent())

    # Everything from here on has to hand the process back or kill it. Ctrl+C during
    # the wait is the normal case, not an edge case -- a 12G checkpoint takes minutes
    # to load -- and if this raises before returning, the caller has no handle to
    # tear down, so the server finishes loading and squats on the port forever.
    try:
        _wait_tcp(wait_host, port, wait_s, proc=proc)

        # start_new_session makes proc the session leader, so everything it forks
        # (lingbot's torchrun workers) shares its sid. A listener outside that
        # session is somebody else's server that beat us to the port.
        # LingBot's torchrun wrapper can create a nested session, so accept
        # descendants of proc too; an unrelated listener still fails both checks.
        holder, argv = _port_holder(port)
        if (holder is not None and _sid(holder) != proc.pid
                and not _is_descendant(holder, proc.pid)):
            raise RuntimeError(
                f"{name} server came up on :{port} but the socket belongs to pid={holder} "
                f"({_show(argv)}), which is not in the session we just started (pid={proc.pid}). "
                f"Refusing to evaluate against an unidentified server.")
    except BaseException:  # includes KeyboardInterrupt
        logger.warning(f"[eval] {name} server startup aborted; killing pid={proc.pid}")
        _kill_server(proc, logger)
        raise
    logger.info(f"[eval] {name} server ready on :{port} (pid={holder or proc.pid})")
    return proc


def _kill_server(proc, logger=None) -> None:
    """Tear down a server started by ``_spawn_server`` -- the whole session, not just it.

    ``start_new_session`` makes the server a session leader, so signalling only the
    direct child leaves anything it forked (lingbot runs under torchrun) alive and
    still holding the port. A leftover listener is not a tidiness problem: the next
    eval attaches to it and reports the wrong checkpoint's numbers.
    """
    if proc is None or proc.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            proc.wait(timeout=10)
            return
        except (ProcessLookupError, PermissionError):
            return
        except Exception:
            continue
    if logger is not None and proc.poll() is None:
        logger.warning(f"[eval] policy server pid={proc.pid} survived SIGKILL; "
                       f"check for a leftover listener before the next run")


def _thirdparty(name: str) -> str:
    """Absolute path to a vendored submodule under ``thirdparty/`` (repo-root relative)."""
    return str(Path(__file__).resolve().parents[4] / "thirdparty" / name)


def _launch_lingbot_server(args, logger):
    launcher = getattr(args, "lingbot_server_launcher", None)
    server_python = getattr(args, "lingbot_server_python", None)
    if not launcher and not server_python:
        return _attach("lingbot", args.lingbot_port, logger)
    # Default to the vendored fork submodule when a repo wasn't explicitly given.
    repo = getattr(args, "lingbot_server_repo", None) or _thirdparty("lingbot-va")

    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"  # shared-box user-site CPU torch shadows conda lingbot-va env (same fix as gr00t/fastwam servers)
    if getattr(args, "lingbot_cuda_visible_devices", None):
        env["CUDA_VISIBLE_DEVICES"] = str(args.lingbot_cuda_visible_devices)
    if getattr(args, "lingbot_config_name", None):
        env["CONFIG_NAME"] = str(args.lingbot_config_name)
    env["PORT"] = str(args.lingbot_port)
    if getattr(args, "lingbot_master_port", None):
        env["MASTER_PORT"] = str(args.lingbot_master_port)
    if getattr(args, "lingbot_conda_env", None):
        env["CONDA_ENV"] = str(args.lingbot_conda_env)
    if getattr(args, "lingbot_conda_base", None):
        env["CONDA_BASE"] = str(args.lingbot_conda_base)
    env["REPO"] = str(repo)
    # LingBot checkpoints contain only the fine-tuned transformer plus symlinked
    # base components.  Treat benchmark.checkpoint as the model root, just like
    # the other policies, and make the matching dataset statistics explicit.
    if getattr(args, "checkpoint", None):
        env["LINGBOT_SILK_MODEL_DIR"] = str(args.checkpoint)
    if getattr(args, "lingbot_dataset_dir", None):
        env["LINGBOT_SILK_DATASET_DIR"] = str(args.lingbot_dataset_dir)
    if getattr(args, "lingbot_save_root", None):
        env["save_root"] = str(args.lingbot_save_root)

    if launcher:
        cwd = repo
        cmd = ["bash", str(launcher)]
    else:
        cwd = str(repo)
        cmd = [
            str(server_python), "-m", "torch.distributed.run",
            "--nproc_per_node", "1",
            "--master_port", str(getattr(args, "lingbot_master_port", 29571)),
            "evaluation/tcp_policy_server.py",
            "--config-name", str(getattr(args, "lingbot_config_name", "silk_infer")),
            "--port", str(args.lingbot_port),
        ]
        if getattr(args, "lingbot_save_root", None):
            cmd += ["--save_root", str(args.lingbot_save_root)]

    return _spawn_server("lingbot", cmd, host=args.lingbot_host, port=args.lingbot_port,
                         wait_s=float(getattr(args, "lingbot_server_wait_s", 420.0)),
                         logger=logger, cwd=cwd, env=env)


def _launch_gr00t_server(args, logger):
    """Auto-spawn the GR00T ZMQ server in its own conda env (py3.10), so a matrix
    run manages it per-cell (killed in the orchestrator ``finally``) exactly like
    lerobot_remote/lingbot. When ``--gr00t_server_python`` is omitted, return None
    and connect to an already-running server (legacy manual mode)."""
    server_python = getattr(args, "gr00t_server_python", None)
    if not server_python:
        return _attach("gr00t", args.gr00t_port, logger)
    repo = getattr(args, "gr00t_server_repo", None)
    if not repo:
        raise ValueError("--gr00t_server_python requires --gr00t_server_repo (Isaac-GR00T dir)")
    modality = getattr(args, "gr00t_modality_config", None) \
        or str(Path(repo) / "examples" / "DeformableBimanual" / "dual_piper_config.py")

    env = dict(os.environ)
    # user-site CPU torch on shared boxes shadows the env's CUDA torch -> force env-only.
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if getattr(args, "gr00t_cuda_visible_devices", None):
        env["CUDA_VISIBLE_DEVICES"] = str(args.gr00t_cuda_visible_devices)

    # When the server is pinned to one GPU via CUDA_VISIBLE_DEVICES, only that
    # card is visible (as cuda:0) — passing the sim's physics device (e.g. cuda:1)
    # would be an invalid ordinal. Use "cuda" in that case (matches fastwam).
    gr00t_device = "cuda" if getattr(args, "gr00t_cuda_visible_devices", None) else (getattr(args, "device", None) or "cuda")
    cmd = [str(server_python), "-m", "gr00t.eval.run_gr00t_server",
           "--model-path", args.checkpoint,
           "--embodiment-tag", "new_embodiment",
           "--modality-config-path", str(modality),
           "--device", gr00t_device,
           "--port", str(args.gr00t_port)]
    return _spawn_server("gr00t", cmd, host=args.gr00t_host, port=args.gr00t_port,
                         wait_s=float(getattr(args, "gr00t_server_wait_s", 420.0)),
                         logger=logger, cwd=str(repo), env=env)


def _launch_lerobot_server(args, logger):
    """Auto-spawn ``serve_lerobot.py`` in a lerobot env of its own (``--policy_type
    lerobot_remote``) for the REAL-ROBOT entry point, which had no wiring for this
    backend at all. Matches the other launchers here: waits for the port,
    ``start_new_session`` for group-kill, and the offline/user-site env guards.

    ``eval_sim`` deliberately keeps its own inline spawn (it declares ``--server_*``
    at its own top level, and its ``Popen`` inherits the caller's CWD, which is
    enough for the symlink trick below) — so nothing on the already-validated sim
    path changes.

    ``--server_cwd`` matters for GR00T-N1.7 under lerobot >= 0.6.0: its processor
    hardcodes the backbone as the *hub id* ``nvidia/Cosmos-Reason2-2B``
    (``policies/groot/processor_groot.py``) and builds the tokenizer from it, so on
    an offline box the server must run from a directory holding an
    ``nvidia/Cosmos-Reason2-2B`` symlink — ``from_pretrained`` checks ``isdir()``
    first and then never reaches the hub. Harmless for every other policy.
    """
    server_python = getattr(args, "server_python", None)
    if not server_python:
        return _attach("lerobot_remote", args.server_port, logger)
    if not getattr(args, "checkpoint", None):
        raise ValueError("lerobot_remote auto-spawn needs --server_python and --checkpoint")

    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"  # shared-box user-site torch would shadow the env
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    if getattr(args, "server_cuda_visible_devices", None):
        env["CUDA_VISIBLE_DEVICES"] = str(args.server_cuda_visible_devices)

    serve_py = str(Path(__file__).parent / "serve_lerobot.py")
    cmd = [str(server_python), serve_py,
           "--checkpoint", str(args.checkpoint),
           "--port", str(args.server_port),
           "--device", getattr(args, "device", None) or "cuda"]
    if getattr(args, "n_action_steps", None) is not None:
        cmd += ["--n_action_steps", str(args.n_action_steps)]

    cwd = getattr(args, "server_cwd", None) or None
    return _spawn_server("lerobot_remote", cmd,
                         host=getattr(args, "server_host", "127.0.0.1"),
                         port=args.server_port,
                         wait_s=float(getattr(args, "server_wait_s", 420.0)),
                         logger=logger, cwd=cwd, env=env)


def _launch_fastwam_server(args, logger):
    """Auto-spawn the FastWAM TCP server in the ``fastwam_eval`` conda env (CWD =
    FastWAM repo root; killed per-cell in the orchestrator ``finally``). When
    ``--fastwam_server_python`` is omitted, return None (connect to a running one)."""
    server_python = getattr(args, "fastwam_server_python", None)
    if not server_python:
        return _attach("fastwam", args.fastwam_port, logger)
    home = getattr(args, "fastwam_home", None) or _thirdparty("FastWAM")
    run_dir = getattr(args, "fastwam_run_dir", None) or args.checkpoint
    if not run_dir:
        raise ValueError("fastwam auto-spawn needs --fastwam_server_python and "
                         "--fastwam_run_dir (or --checkpoint)")

    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"  # shared-box user-site torch would shadow the env
    env["FASTWAM_HOME"] = str(home)
    if getattr(args, "fastwam_cuda_visible_devices", None):
        env["CUDA_VISIBLE_DEVICES"] = str(args.fastwam_cuda_visible_devices)

    server_py = str(Path(__file__).parent / "fastwam" / "server.py")
    cmd = [str(server_python), server_py,
           "--port", str(args.fastwam_port),
           "--run_dir", str(run_dir),
           "--fastwam_home", str(home),
           "--ckpt_step", str(getattr(args, "fastwam_ckpt_step", 5835)),
           "--device", "cuda",
           "--action_horizon", str(getattr(args, "fastwam_action_horizon", 32)),
           "--num_inference_steps", str(getattr(args, "fastwam_num_inference_steps", 20)),
           "--replan_steps", str(getattr(args, "fastwam_replan_steps", 32)),
           "--seed", str(getattr(args, "fastwam_seed", 0))]
    if getattr(args, "fastwam_text_cache", None):
        cmd += ["--text_embeds_cache_dir", str(args.fastwam_text_cache)]
    return _spawn_server("fastwam", cmd, host=args.fastwam_host, port=args.fastwam_port,
                         wait_s=float(getattr(args, "fastwam_server_wait_s", 600.0)),
                         logger=logger, cwd=str(home), env=env)
