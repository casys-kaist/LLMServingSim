"""Rich-backed logger for the simulator.

All simulator output funnels through this module — log records, per-
interval status dashboards, result summaries, and the startup banner.
The public API (``configure_logger``, ``get_logger``, and the
``ComponentLoggerAdapter`` returned by ``get_logger``) is preserved
for every existing call site; the change is cosmetic (Rich colours,
markup, and tracebacks) rather than structural.

Line format kept from the original ANSI implementation:

    [14:47:13.208] [Scheduler] [node=0,inst=1] INFO  iteration 0 finished

- timestamp is ``HH:MM:SS.mmm`` by default (file-handler keeps the
  full ``YYYY-MM-DD HH:MM:SS.mmm`` for offline log parsing);
- component / node / instance are injected by the adapter;
- level is styled by a shared Rich theme so file output stays plain.

Extras:

- ``is_summary=True`` on a log record emits the message verbatim
  (no timestamp / prefix). Used for result summaries.
- ``success()`` emits a green check-mark line at INFO level.
- ``stage(title)`` / ``progress(label, total)`` context managers
  surface long-running steps in a uniform way, mirroring the
  profiler's helpers.
- ``console`` / ``print_rule`` / ``print_banner`` expose Rich primitives
  so callers can render panels and rules without reaching for ``rich``
  directly.
"""
from __future__ import annotations

import logging
import textwrap
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Optional, Union

from rich.align import Align
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.theme import Theme


PROJECT_ROOT_LOGGER_NAME = "llmservingsim"

# ---------------------------------------------------------------------------
# Shared Rich console + theme
# ---------------------------------------------------------------------------

_THEME = Theme(
    {
        "logging.level.debug": "dim cyan",
        "logging.level.info": "cyan",
        "logging.level.warning": "yellow",
        "logging.level.error": "red",
        "logging.level.critical": "red reverse",
        "sim.time": "dim",
        "sim.component": "bold cyan",
        "sim.tag": "magenta",
        "sim.rule": "bold magenta",
        "sim.heading": "bold cyan",
        "sim.tagline": "magenta",
        "ok": "green bold",
    }
)
# One console for everything. Stdout so ``python -m serving ... > out.log``
# captures logs with a single ``>`` (no ``2>&1``). ASTRA-Sim IPC in
# controller.py uses the subprocess's own pipes, not the parent's
# stdout, so this is safe.
#
#  - ``soft_wrap=True``: long log lines don't get chopped; Rich emits
#    them on a single logical line regardless of terminal width.
#  - **No** ``force_terminal``: Rich auto-detects. Interactive
#    terminals get ANSI colours; redirected files / pipes get plain
#    text. This keeps log files free of stray escape codes so the
#    editor/viewer can apply its own log-pattern highlighting. If an
#    IDE's integrated terminal doesn't self-identify as a TTY and
#    colours vanish there, set ``FORCE_COLOR=1`` in the environment
#    and Rich will flip force_terminal on for that session.
#
# No fixed width — rules, centred titles, and the logo all size off
# the current terminal width via ``rich.align.Align`` / ``justify=
# "center"`` / ``console.rule``. This means the banner naturally
# grows with the user's window instead of being pinned to 80.
_console = Console(
    theme=_THEME,
    soft_wrap=True,
)

_LEVEL_STYLES = {
    logging.DEBUG: "logging.level.debug",
    logging.INFO: "logging.level.info",
    logging.WARNING: "logging.level.warning",
    logging.ERROR: "logging.level.error",
    logging.CRITICAL: "logging.level.critical",
}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


class _RichSimHandler(logging.Handler):
    """Rich-backed handler that renders our ``[time] [Component] [tag] LEVEL msg``
    shape. Falls back to verbatim output when ``record.is_summary`` is set.
    """

    def __init__(self, console: Console) -> None:
        super().__init__()
        self._console = console

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            msg = record.getMessage()
        except Exception:
            self.handleError(record)
            return

        # Summary mode: message only, no decoration.
        if getattr(record, "is_summary", False):
            self._console.print(msg, markup=True, highlight=False)
            if record.exc_info:
                self._console.print_exception()
            return

        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        ms = int(record.msecs)
        component = escape(str(getattr(record, "component", record.name)))

        node_id = getattr(record, "node_id", None)
        instance_id = getattr(record, "instance_id", None)
        # Bracket characters live INSIDE the styled span so the
        # rendered line starts with an ANSI escape (byte 0x1b) —
        # editors / log viewers that auto-detect ANSI colouring key
        # on that first byte to decide whether to render codes or
        # leave them literal.
        if node_id is not None and instance_id is not None:
            tag_seg = f" [sim.tag]\\[node={node_id},inst={instance_id}][/]"
        elif node_id is not None:
            tag_seg = f" [sim.tag]\\[node={node_id}][/]"
        elif instance_id is not None:
            tag_seg = f" [sim.tag]\\[inst={instance_id}][/]"
        else:
            tag_seg = ""

        style = _LEVEL_STYLES.get(record.levelno, "logging.level.info")
        level = record.levelname.upper()
        # Opt-in markup for callers who pass pre-styled strings (the
        # adapter's success() helper does). Default is escape-everything
        # so log messages containing `[stuff]` never accidentally render.
        body = msg if getattr(record, "_rich_markup", False) else escape(msg)

        self._console.print(
            f"[sim.time]\\[{ts}.{ms:03d}][/] "
            f"[sim.component]\\[{component}][/]{tag_seg} "
            f"[{style}]{level:<8}[/] {body}",
            markup=True,
            highlight=False,
        )
        if record.exc_info:
            self._console.print_exception()


class _PlainFileFormatter(logging.Formatter):
    """Plain-text formatter for optional file output — no ANSI, no Rich."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        dt = datetime.fromtimestamp(record.created)
        return f"{dt:%Y-%m-%d %H:%M:%S}.{int(record.msecs):03d}"

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "is_summary", False):
            line = record.getMessage()
            if record.exc_info and not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                line += "\n" + record.exc_text
            return line

        ts = self.formatTime(record)
        component = getattr(record, "component", record.name)
        node_id = getattr(record, "node_id", None)
        instance_id = getattr(record, "instance_id", None)
        if node_id is not None and instance_id is not None:
            tag = f" [node={node_id},inst={instance_id}]"
        elif node_id is not None:
            tag = f" [node={node_id}]"
        elif instance_id is not None:
            tag = f" [inst={instance_id}]"
        else:
            tag = ""
        line = f"[{ts}] [{component}]{tag} [{record.levelname}] {record.getMessage()}"
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            line += "\n" + record.exc_text
        return line


# ---------------------------------------------------------------------------
# Adapter — injects component / node / instance onto every record
# ---------------------------------------------------------------------------


class ComponentLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter that attaches ``component`` / ``node_id`` / ``instance_id``
    to every record the wrapped logger emits.

    All five standard log methods are available (debug / info / warning /
    error / critical). Also exposes ``success()`` (green ✓ tag at INFO
    level) and ``summary()`` (prints the message verbatim, no prefix).
    """

    def __init__(
        self,
        logger: logging.Logger,
        component: str,
        node_id: Optional[int] = None,
        instance_id: Optional[int] = None,
    ) -> None:
        super().__init__(logger, extra={})
        self.component = component
        self.node_id = node_id
        self.instance_id = instance_id

    def process(self, msg: Any, kwargs: dict) -> tuple[Any, dict]:
        extra = kwargs.get("extra") or {}
        extra.setdefault("component", self.component)
        extra.setdefault("node_id", self.node_id)
        extra.setdefault("instance_id", self.instance_id)
        kwargs["extra"] = extra
        return msg, kwargs

    # Convenience extensions ------------------------------------------------

    def success(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Emit a green ✓-marked line at INFO level."""
        extra = kwargs.pop("extra", None) or {}
        extra["_rich_markup"] = True
        kwargs["extra"] = extra
        self.info("[ok]✓[/ok] " + msg, *args, **kwargs)

    def summary(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Emit the message without prefix/timestamp decorations."""
        extra = kwargs.pop("extra", None) or {}
        extra["is_summary"] = True
        kwargs["extra"] = extra
        self.info(msg, *args, **kwargs)


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

_configured = False


def configure_logger(
    level: Union[str, int] = "INFO",
    *,
    log_file: Optional[str] = None,
) -> None:
    """Initialize the project logger.

    Idempotent: second call just updates the level on the already-
    attached handlers. File output is plain text (no ANSI / Rich
    markup) so `grep`/`less` on a saved log behaves sanely.
    """
    global _configured
    root = logging.getLogger(PROJECT_ROOT_LOGGER_NAME)

    if isinstance(level, str):
        lvl = getattr(logging, level.upper(), logging.INFO)
    else:
        lvl = int(level)

    if _configured:
        root.setLevel(lvl)
        for handler in root.handlers:
            handler.setLevel(lvl)
        return

    root.handlers.clear()
    root.setLevel(lvl)

    console_handler = _RichSimHandler(_console)
    console_handler.setLevel(lvl)
    root.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(lvl)
        file_handler.setFormatter(_PlainFileFormatter())
        root.addHandler(file_handler)

    root.propagate = False
    _configured = True


def get_logger(
    component: Optional[Union[str, type]] = None,
    *,
    node_id: Optional[int] = None,
    instance_id: Optional[int] = None,
) -> ComponentLoggerAdapter:
    """Return a ``ComponentLoggerAdapter`` bound to this project's root.

    Accepts a class (uses ``__name__``) or a free-form string as the
    component label. ``node_id`` / ``instance_id`` are optional tags
    that flow into every record.
    """
    if component is None:
        component_name = "Global"
    elif isinstance(component, str):
        component_name = component
    else:
        component_name = getattr(component, "__name__", str(component))

    base_logger = logging.getLogger(PROJECT_ROOT_LOGGER_NAME)
    return ComponentLoggerAdapter(
        base_logger,
        component=component_name,
        node_id=node_id,
        instance_id=instance_id,
    )


# ---------------------------------------------------------------------------
# Rich console primitives re-exported for callers
# ---------------------------------------------------------------------------


def console() -> Console:
    """Return the shared stderr-bound Rich Console."""
    return _console


def print_rule(label: str = "", style: str = "sim.rule") -> None:
    """Draw a horizontal rule across the current terminal width,
    optionally with a centred label.
    """
    _console.rule(label, style=style)


def print_heading(title: str) -> None:
    """Centre a section title with no rule characters around it.

    For a section that opens under a rule already drawn by whatever printed
    before it -- ``print_rule`` would draw a second full-width line, leaving two
    stacked rules with the title buried in the lower one.
    """
    _console.print(f"[sim.tagline]{title}[/]", justify="center")


def print_markup(message: str) -> None:
    """Print a line of Rich markup without any logging decoration.

    Uses the shared soft-wrapping console so long messages don't get
    chopped into multiple lines by Rich.
    """
    _console.print(message, markup=True, highlight=False)


# ---------------------------------------------------------------------------
# High-level display helpers
# ---------------------------------------------------------------------------


LLMSERVINGSIM_LOGO = r"""     _    _    __  __ ___              _           ___ _       ___   __
    | |  | |  |  \/  / __| ___ _ ___ _(_)_ _  __ _/ __(_)_ __ |_  ) /  \
    | |__| |__| |\/| \__ \/ -_) '_\ V / | ' \/ _` \__ \ | '  \ / / | () |
    |____|____|_|  |_|___/\___|_|  \_/|_|_||_\__, |___/_|_|_|_/___(_)__/
                                             |___/       """


def _centered_logo() -> Align:
    """Return a Rich renderable that centres the multi-line ASCII
    logo as a single block (preserving internal column alignment)
    within the current console width. Using ``Align.center`` instead
    of hand-padded strings means the logo re-centres automatically
    when the terminal is resized or has a different width than 80.
    """
    stripped = textwrap.dedent(LLMSERVINGSIM_LOGO)
    return Align.center(Text(stripped, style="sim.heading"))


def print_banner() -> None:
    """Render the startup banner — title, tagline, ASCII logo.

    Everything sizes off the current terminal width — rules fill it,
    the tagline centres in it, the logo block centres in it. On
    wider terminals the banner grows; on narrower ones it stays snug.
    """
    title = "LLMServingSim2.0"
    tagline = (
        "A Unified Simulator for Heterogeneous Hardware "
        "and Serving Techniques in LLM"
    )
    _console.rule(f"[sim.heading]{title}[/]", style="sim.rule")
    _console.print(f"[sim.tagline]{tagline}[/]", justify="center")
    _console.rule(style="sim.rule")
    _console.print(_centered_logo())
    _console.rule(style="sim.rule")


def print_input_config(args: Any) -> None:
    """Render the argparse-namespace configuration block used at startup."""
    def _inf0(x: Any) -> Any:
        return x if x not in (0, None) else "inf"

    def _bits(x: Any) -> str:
        return f"{x}-bit" if x is not None else "N/A"

    def _yn(x: Any) -> str:
        return "ENABLED" if x else "DISABLED"

    def _na(x: Any) -> Any:
        return x if x not in (None, "") else "N/A"

    def _pc(x: Any) -> str:
        if x == "None":
            return "xPU-Only"
        if x == "CPU":
            return "xPU + CPU"
        if x == "CXL":
            return "xPU + CXL"
        return "None"

    def _derived(x: Any) -> str:
        # ``--block-size`` defaults to None, meaning "whatever the profile
        # bundle records vLLM settling on". That is resolved per instance, so
        # the banner cannot name one number -- but it must not print "None"
        # either, which reads as "unset" rather than "derived".
        return "derived per instance" if x is None else x

    items: list[tuple[str, Any]] = []

    def add(attr: str, label: str, conv=lambda v: v) -> None:
        if hasattr(args, attr):
            items.append((label, conv(getattr(args, attr))))

    add("cluster_config", "Cluster config", _na)
    add("run_id", "Run ID", _na)
    add("inputs_root", "ASTRA-Sim inputs root", _na)
    add("memory_config", "Memory config", _na)
    add("dataset", "Dataset", _na)
    add("num_req", "Num requests")
    add("max_num_seqs", "Max num seqs", _inf0)
    add("max_num_batched_tokens", "Max batched tokens", _inf0)
    add("block_size", "Block size (tokens)", _derived)
    add("fp", "FP precision", _bits)
    add("request_routing_policy", "Request routing", _na)
    add("expert_routing_policy", "Expert routing", _na)
    add("enable_prefix_caching", "Prefix caching", _yn)
    add("enable_chunked_prefill", "Chunked prefill", _yn)
    add("prefix_storage", "Prefix caching scheme", _pc)
    add("enable_prefix_sharing", "Centralized prefix caching", _yn)
    add("enable_attn_offloading", "Offload attention to PIM", _yn)
    add("enable_sub_batch_interleaving", "Sub-batch interleaving", _yn)
    add("enable_attn_prediction", "Realtime attention prediction", _yn)
    add("link_bw", "Link bandwidth (GB/s)")
    add("link_latency", "Link latency (ns)")
    add("network_backend", "Network backend", _na)
    add("log_interval", "Log interval (s)")
    add("log_level", "Log level", _na)

    # Heading centres to the current terminal width (so it lines up
    # under whatever the banner rules spanned). Per-item lines stay
    # left-aligned so long paths / dataset names flow naturally.
    _console.print("[sim.tagline]Input configuration[/]", justify="center")
    if not items:
        _console.print("[dim]  (no parsed arguments to display)[/]")
        _console.rule(style="sim.rule")
        return
    _console.print()
    key_pad = max(len(k) for k, _ in items)
    for key, val in items:
        _console.print(f"  • [cyan]{key:<{key_pad}}[/cyan] : {val}")
    _console.rule(style="sim.rule")


@contextmanager
def stage(title: str) -> Iterator[None]:
    """Bracket a discrete step with ``→ title`` / green ``✓ title (Ns)``.

    Use for startup steps (config load, ASTRA-Sim spawn) or wind-down
    stages where the eye wants a success/failure indicator.
    """
    import time as _t

    _console.print(f"[sim.time]→[/] {title}")
    t0 = _t.monotonic()
    try:
        yield
    except Exception:
        _console.print(
            f"[logging.level.error]✗[/] {title} "
            f"([dim]failed after {_t.monotonic() - t0:.1f}s[/])"
        )
        raise
    _console.print(f"[ok]✓[/ok] {title} ([dim]{_t.monotonic() - t0:.1f}s[/])")


class _Bar:
    """Thin handle around rich's ``TaskID`` so callers don't import rich."""

    def __init__(self, progress: Progress, task_id: Any) -> None:
        self._progress = progress
        self._task_id = task_id

    def advance(self, n: int = 1) -> None:
        self._progress.update(self._task_id, advance=n)

    def update_total(self, total: int) -> None:
        self._progress.update(self._task_id, total=total)


@contextmanager
def progress(label: str, total: int) -> Iterator[_Bar]:
    """Yield a bar tied to a live rich progress UI.

    Falls back to plain periodic updates on non-TTY streams.
    """
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
        yield _Bar(prog, task_id)


__all__ = [
    "PROJECT_ROOT_LOGGER_NAME",
    "LLMSERVINGSIM_LOGO",
    "configure_logger",
    "get_logger",
    "ComponentLoggerAdapter",
    "console",
    "print_rule",
    "print_heading",
    "print_markup",
    "print_banner",
    "print_input_config",
    "stage",
    "progress",
]


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logger(level="DEBUG")
    print_banner()
    log = get_logger("Demo", node_id=0, instance_id=1)
    log.info("iteration 0 finished, exposed communication 0 cycles.")
    log.debug("debug info for scheduling decision.")
    log.warning("KV cache usage above 80%.")
    log.error("ASTRA-Sim pipe read timeout.")
    log.success("simulation wrapped up")
    log.summary("TTFT mean: 7.71 s  |  TPOT mean: 55.8 ms")
    with stage("example work"):
        import time as _t
        _t.sleep(0.2)
    with progress("cooking", total=3) as bar:
        for _ in range(3):
            import time as _t
            _t.sleep(0.1)
            bar.advance()
