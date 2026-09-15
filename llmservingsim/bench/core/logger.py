"""Logger + progress UI for the bench module.

Mirrors the shape of ``profiler.core.logger`` so the three modules
(profiler / bench / serving) share the same look and call patterns.
The rest of the bench package is forbidden from calling ``print()``
directly — every user-facing line goes through this module.

Conventions:

* Built on stdlib ``logging`` so verbosity is just a log level.
* Uses ``rich`` for colourised console output, soft-wrapped lines,
  and progress bars. ``rich`` ships with vLLM as a transitive
  dependency, so no extra install.
* ``capture_stdio()`` redirects C-level stdout/stderr during vLLM
  engine boot — vLLM's worker processes print from C++ / CUDA init
  which bypasses Python streams.

Callers should ``from bench.core import logger as log`` so call sites
read ``log.info(...)``, ``log.success(...)``, ``log.stage(...)``, etc.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.panel import Panel
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


# --- module-level singletons ------------------------------------------------

_THEME = Theme({
    "logging.level.info":    "cyan",
    "logging.level.warning": "yellow",
    "logging.level.error":   "red",
    "logging.level.debug":   "dim",
    "ok":                    "green bold",
    "bench.rule":            "magenta",
    "bench.banner":          "magenta bold",
})

# stdout, soft-wrapped, no forced terminal — Rich auto-detects TTY.
_console = Console(theme=_THEME, soft_wrap=True)

_logger = logging.getLogger("bench")
_configured = False


# --- public API -------------------------------------------------------------

def configure(level: int | str = logging.INFO) -> None:
    """Initialize the bench logger.

    Idempotent: re-invoking just updates the level on the already-
    attached handler. vLLM's loggers are clamped to ERROR unless we're
    at DEBUG so its startup banner doesn't drown bench output.
    """
    global _configured

    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if not isinstance(resolved, int):
            raise ValueError(f"Unknown log level: {level!r}")
        level = resolved

    _logger.setLevel(level)

    if not _configured:
        from rich.logging import RichHandler
        handler = RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            log_time_format="[%H:%M:%S]",
        )
        _logger.addHandler(handler)
        _logger.propagate = False
        _configured = True

    vllm_level = logging.DEBUG if level <= logging.DEBUG else logging.ERROR
    for name in ("vllm", "vllm.engine", "vllm.worker", "vllm.executor",
                 "vllm.config", "vllm.model_executor", "vllm.distributed"):
        logging.getLogger(name).setLevel(vllm_level)


@contextmanager
def capture_stdio() -> Iterator[None]:
    """Redirect C-level fd 1 and 2 to a tmpfile during the ``with`` block.

    vLLM's engine startup prints from C++ (pybind11 / torch) and CUDA
    library init — those bypass ``sys.stdout``, so contextlib.redirect_stdout
    is insufficient. On exception inside the block, the captured output is
    re-emitted at ERROR for post-mortem.

    No-op at DEBUG verbosity (user wants raw vLLM noise).
    """
    if _logger.isEnabledFor(logging.DEBUG):
        yield
        return

    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    buf = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(buf.fileno(), 1)
        os.dup2(buf.fileno(), 2)
        try:
            yield
        except Exception:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            buf.seek(0)
            captured = buf.read().decode(errors="replace")
            if captured.strip():
                _logger.error("Captured vLLM stdio before failure:\n%s",
                              captured)
            raise
    finally:
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
    """INFO-level line tagged with a green check-mark."""
    _logger.info("[ok]✓[/ok] " + msg, *args, **kw)


# --- high-level display helpers --------------------------------------------

def print_rule(label: str = "") -> None:
    """Render a Rich rule (horizontal divider) on the console."""
    _console.rule(label, style="bench.rule")


def print_banner(title: str = "bench", subtitle: str | None = None) -> None:
    """Render a one-line bench banner."""
    panel = Panel(
        f"[bench.banner]{title}[/bench.banner]"
        + (f"\n{subtitle}" if subtitle else ""),
        border_style="bench.rule",
        expand=False,
    )
    _console.print(panel)


@contextmanager
def stage(title: str) -> Iterator[None]:
    """Bracket a long step with start/end markers + elapsed time.

    Example::

        with log.stage("Booting AsyncLLM"):
            engine = AsyncLLM.from_engine_args(args)
    """
    info("[bold]%s[/bold] …", title)
    t0 = time.monotonic()
    try:
        yield
    except Exception:
        error("[bold]%s[/bold] failed after %.1fs", title, time.monotonic() - t0)
        raise
    info("[ok]✓[/ok] [bold]%s[/bold] (%.1fs)", title, time.monotonic() - t0)


@contextmanager
def progress(label: str, total: int) -> Iterator["_Bar"]:
    """Render a Rich progress bar for a known-total operation.

    The yielded ``_Bar`` exposes one method, ``advance(n=1)``, so call
    sites don't need to track the underlying task id.
    """
    columns = (
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=_console, transient=False) as bar:
        task = bar.add_task(label, total=total)
        yield _Bar(bar, task)


class _Bar:
    """Tiny adapter around rich.progress.Progress + a task id."""

    def __init__(self, bar: Progress, task: int) -> None:
        self._bar = bar
        self._task = task

    def advance(self, n: int = 1) -> None:
        self._bar.advance(self._task, n)


# Expose the underlying console for callers that need to print Panels /
# Tables / etc directly.
console = _console
