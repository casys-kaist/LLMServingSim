"""Logger + progress UI for the profiler.

All user-facing output goes through this module. The rest of the
package is forbidden from calling print() directly — this gives us one
place to control verbosity, colorize output, capture vLLM's noisy
stdout/stderr, and render progress bars.

Design decisions:

* Built on top of Python's stdlib ``logging`` so verbosity is just a
  standard log level. ``--log-level INFO`` and ``-v`` both resolve to
  the same ``logger.configure(logging.INFO)`` call.

* Uses ``rich`` for progress bars, console colorization, and
  traceback pretty-printing. ``rich`` is already pulled in as a vLLM
  dependency so we add no new install burden.

* vLLM itself logs a ~50-line banner on every ``LLM()`` construction.
  We silence vLLM's own loggers below ERROR by default; at DEBUG we
  let them through for troubleshooting.

* ``capture_stdio()`` redirects C-level stdout/stderr (not just
  Python's ``sys.stdout``) during vLLM engine construction, because
  vLLM's worker processes print via CUDA libraries and torch C++ that
  bypass Python streams. On failure the captured buffer is dumped to
  ERROR for post-mortem.

Callers should conventionally ``from profiler.core import logger as log``
so call sites read ``log.info(...)``, ``log.progress(...)``, etc.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

if TYPE_CHECKING:
    # Avoid import cycle: config.py imports nothing from logger, but
    # logger wants ProfileArgs for banner()'s type hint only.
    from llmservingsim.profiler.core.config import ProfileArgs


# --- module-level singletons ------------------------------------------------

# Console streams to stdout so ``python -m llmservingsim.profiler profile ... > out.log``
# captures the full output with a single ``>``. CSV writers do their own
# file I/O (not stdout), so the profiler's own stdout is free for logs.
_THEME = Theme(
    {
        "logging.level.info": "cyan",
        "logging.level.warning": "yellow",
        "logging.level.error": "red",
        "logging.level.debug": "dim",
        # Custom tag used by success() — mapped via markup, not a real level.
        "ok": "green bold",
    }
)
# ``soft_wrap=True`` tells Rich not to hard-wrap a log line when its
# rendered length exceeds the terminal width. Without this, messages
# like multi-path skew grids or long skew_fit keys get chopped across
# several visual lines.
#
# No ``force_terminal`` — Rich auto-detects. Interactive terminals get
# ANSI colours; redirected files (``> out.log`` / nohup) get plain
# text, which means no stray escape bytes end up in the log file and
# IDE viewers can run their own log-pattern highlighting. If an IDE
# terminal doesn't self-identify as a TTY and colours vanish there,
# set ``FORCE_COLOR=1`` in the environment.
_console = Console(theme=_THEME, soft_wrap=True)

# The single logger the rest of the profiler uses. Children of this
# logger (e.g. logging.getLogger("profiler.runner")) inherit its
# handlers and level automatically.
_logger = logging.getLogger("profiler")

# Guard so configure() is idempotent — re-invocation from tests or nested
# commands shouldn't attach multiple handlers.
_configured = False


# --- public API -------------------------------------------------------------

def configure(level: int | str = logging.WARNING) -> None:
    """Initialize the profiler logger.

    Accepts either a stdlib level constant (``logging.INFO``) or the
    stringy form (``"INFO"``). The CLI layer in ``__main__.py``
    translates ``-v`` / ``-vv`` / ``--log-level`` into this call.

    Side effects:
        * Attaches a RichHandler to the "profiler" logger (once).
        * Suppresses vLLM's own loggers unless level is DEBUG.
    """
    global _configured

    # Normalize the level argument to an int.
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            raise ValueError(f"Unknown log level: {level!r}")

    _logger.setLevel(level)

    # Attach the RichHandler only once. Re-configuring just updates the
    # level on the already-attached handler.
    if not _configured:
        handler = RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            log_time_format="[%H:%M:%S]",
        )
        _logger.addHandler(handler)
        # Do not propagate to the root logger; otherwise every log line
        # appears twice once the root logger has a handler (common in
        # tests that call logging.basicConfig).
        _logger.propagate = False
        _configured = True

    # Decide how loud vLLM itself is allowed to be.
    # DEBUG → let everything through (user is troubleshooting).
    # Anything quieter → clamp vLLM to ERROR so its startup banner doesn't
    # drown our own output.
    vllm_level = logging.DEBUG if level <= logging.DEBUG else logging.ERROR
    for name in ("vllm", "vllm.engine", "vllm.worker", "vllm.executor",
                 "vllm.config", "vllm.model_executor", "vllm.distributed"):
        logging.getLogger(name).setLevel(vllm_level)


@contextmanager
def capture_stdio() -> Iterator[None]:
    """Capture C-level stdout+stderr while inside the ``with`` block.

    Why this is not just contextlib.redirect_stdout:
        vLLM's engine startup prints from C++ (pybind11 / torch) and
        from CUDA library init. Those bypass ``sys.stdout``; you have
        to swap the underlying file descriptors (1 and 2) to silence
        them. We redirect both to a tempfile, then on exit either
        discard the content (success) or re-emit it at ERROR level
        (failure).

        At DEBUG verbosity we no-op so the user can see vLLM's own
        diagnostics verbatim.
    """
    if _logger.isEnabledFor(logging.DEBUG):
        # User asked for maximal noise; don't hide anything.
        yield
        return

    # Save current fds so we can restore them.
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    # Open a tmpfile to collect captured output.
    import tempfile
    buf = tempfile.TemporaryFile(mode="w+b")
    try:
        # Point fd 1 and 2 at the buffer.
        os.dup2(buf.fileno(), 1)
        os.dup2(buf.fileno(), 2)
        try:
            yield
        except Exception:
            # Surface the captured output so the user can debug.
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            buf.seek(0)
            captured = buf.read().decode(errors="replace")
            if captured.strip():
                _logger.error("Captured vLLM stdio before failure:\n%s",
                              captured)
            raise
    finally:
        # Always restore original fds.
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        buf.close()


# --- convenience wrappers ---------------------------------------------------

def info(msg: str, *args, **kw) -> None:
    _logger.info(msg, *args, **kw)


def warning(msg: str, *args, **kw) -> None:
    _logger.warning(msg, *args, **kw)


def error(msg: str, *args, **kw) -> None:
    _logger.error(msg, *args, **kw)


def debug(msg: str, *args, **kw) -> None:
    _logger.debug(msg, *args, **kw)


def success(msg: str, *args, **kw) -> None:
    """A log line tagged as 'SUCCESS'.

    Python's logging module has no SUCCESS level. We emit at INFO with
    a green marker so it stands out in the terminal without needing a
    custom level number.
    """
    _logger.info("[ok]✓[/ok] " + msg, *args, **kw)


# --- high-level display helpers --------------------------------------------

def banner(args: "ProfileArgs", root: Path) -> None:
    """Print the big "here's what we're about to do" header at run start."""
    _console.rule(
        f"[bold cyan]Profiling {args.model} on {args.hardware}[/] "
        f"([dim]{args.architecture}[/])"
    )
    info("Variant: [bold]%s[/]", args.effective_variant)
    info("TP degrees: %s", args.tp_degrees)
    info("Output root: [dim]%s[/]", root)


def done(root: Path) -> None:
    """Print the closing summary line."""
    _console.rule("[bold green]Done[/]")
    info("Results written to: [bold]%s[/]", root)


@contextmanager
def stage(title: str) -> Iterator[None]:
    """Bracket a discrete step with '→ title' / '✓ title (Ns)'.

    Use for anything that takes more than a second and doesn't warrant
    its own progress bar (engine boot, file IO, etc.).
    """
    info("→ %s", title)
    t0 = time.monotonic()
    try:
        yield
    except Exception:
        error("✗ %s (failed after %.1fs)", title, time.monotonic() - t0)
        raise
    else:
        success("%s (%.1fs)", title, time.monotonic() - t0)


class Bar:
    """Thin handle around rich's ``TaskID`` so callers don't import rich."""

    def __init__(self, progress: Progress, task_id):
        self._progress = progress
        self._task_id = task_id

    def advance(self, n: int = 1) -> None:
        self._progress.update(self._task_id, advance=n)

    def update_total(self, total: int) -> None:
        """Allow the caller to correct the total after the fact
        (grid generators sometimes can't know the feasible count up
        front — see grid filtering in categories.py)."""
        self._progress.update(self._task_id, total=total)


@contextmanager
def progress(label: str, total: int) -> Iterator[Bar]:
    """Context manager yielding a Bar tied to a live rich progress UI.

    If stdout is not a TTY (e.g. redirected to a file) rich
    auto-falls back to periodic plain-text updates, so there's no
    need for a special "no TTY" mode here.
    """
    # Columns chosen to match the layout shown in DESIGN_V2.md §9.3.
    columns = [
        TextColumn("[bold]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]
    with Progress(*columns, console=_console, transient=False) as prog:
        task_id = prog.add_task(label, total=total)
        yield Bar(prog, task_id)
