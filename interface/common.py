"""Helpers shared by the `rag` and `scraper` Gradio apps.

Everything here is intentionally backend-agnostic: pure data/IO utilities and
log-capture plumbing with no dependency on `rag` or `scraper`. App-specific
constants (e.g. each app's ``PREVIEW_COLS``) and flows deliberately stay in
their own modules.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd


# ---------------------------------------------------------------------------
# Dataframe rendering
# ---------------------------------------------------------------------------

def stringify_found_on(v) -> str:
    """Render a ``found_on`` cell as a comma-joined string for ``gr.Dataframe``.

    The cell can be a Python list, a numpy array (pandas may turn list objects
    into ndarrays during concat/merge), ``None``, or NaN. We can't use
    ``v or ""`` because truth-value testing on a numpy array raises ValueError.
    """
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):  # scalar NaN only
        return ""
    if hasattr(v, "tolist"):  # numpy array / pandas Series
        v = v.tolist()
    if isinstance(v, (list, tuple, set)):
        return ", ".join(str(x) for x in v)
    return str(v)


# ---------------------------------------------------------------------------
# Live log capture
# ---------------------------------------------------------------------------

class TeeStream:
    """Write to both the original stream and a StringIO buffer.

    Used with ``contextlib.redirect_stdout`` / ``redirect_stderr`` so a worker
    thread's terminal output can be tailed into a Gradio textbox while still
    reaching the real console.
    """

    def __init__(self, original, buffer: io.StringIO):
        self._original = original
        self._buffer = buffer

    def write(self, data):
        self._original.write(data)
        self._buffer.write(data)

    def flush(self):
        self._original.flush()
        self._buffer.flush()

    # make it a valid stream for libraries that check for these
    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return False


def tail(text: str, n: int = 10) -> str:
    """Return the last ``n`` non-empty lines of ``text``."""
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def run_in_thread_with_log(
    fn: Callable,
    *args,
    tail_lines: int = 10,
    **kwargs,
):
    """Run ``fn(*args, **kwargs)`` in a worker thread, streaming its stdout/
    stderr as a tailed log.

    A generator: yields the last ``tail_lines`` of captured output (a plain
    string) roughly twice a second while the work runs, then *returns* the
    function's return value (PEP 380 ``return``, surfaced as the generator's
    ``StopIteration.value``). Any exception raised inside ``fn`` is re-raised
    here as a ``gr.Error`` once the thread finishes, so callers get Gradio's
    inline error toast rather than a silent failure.

    Backend-agnostic by design: it knows nothing about the caller's Gradio
    output shape. Callers map the yielded log string into their own N-tuple
    (typically ``gr.skip()`` for every non-log slot) and consume the return
    value via the generator's ``StopIteration.value``. See
    ``rag_app._autocode_cv`` for the canonical call pattern.
    """
    import gradio as gr  # local import keeps common.py gradio-free at module load

    buf = io.StringIO()
    result: dict = {}
    exc_holder: dict = {}

    def _worker():
        tee_out = TeeStream(sys.stdout, buf)
        tee_err = TeeStream(sys.stderr, buf)
        try:
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                result["value"] = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 -- surfaced to the user below
            exc_holder["e"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while t.is_alive():
        t.join(timeout=0.5)
        yield tail(buf.getvalue(), tail_lines)

    if "e" in exc_holder:
        raise gr.Error(str(exc_holder["e"]))

    return result["value"]


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def env_warning(env_vars: Mapping[str, str]) -> str:
    """Return a markdown warning listing any unset env vars, or "" if all set.

    ``env_vars`` maps variable name -> human description; the description is
    shown next to each missing variable.
    """
    missing = [
        f"- `{var}`: {desc}"
        for var, desc in env_vars.items()
        if not os.environ.get(var)
    ]
    if not missing:
        return ""
    return (
        "⚠️ **The following environment variables are not set:**\n\n"
        + "\n".join(missing)
        + "\n\nSet them (e.g. in a `.env` file) before running."
    )


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_table_outputs(
    df: pd.DataFrame,
    prefix: str,
    csv_drop_cols: Iterable[str] = (),
) -> tuple[Path, str]:
    """Write ``df`` to a timestamped parquet + csv under a fresh temp dir.

    A new ``tempfile.mkdtemp`` is used per call so concurrent Gradio sessions
    don't stomp on each other. The parquet gets the full frame; the csv drops
    ``csv_drop_cols`` (e.g. heavy ``raw``/``description`` columns).

    Returns ``(out_dir, stamp)`` so callers can write additional artifacts
    (e.g. a raw-JSON zip) into the same directory using the same timestamp.
    The parquet and csv land at ``out_dir / f"{prefix}_{stamp}.parquet"`` and
    ``out_dir / f"{prefix}_{stamp}.csv"``.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}_{stamp}_"))

    parquet_path = out_dir / f"{prefix}_{stamp}.parquet"
    csv_path = out_dir / f"{prefix}_{stamp}.csv"

    df.to_parquet(parquet_path)
    df.drop(columns=list(csv_drop_cols), errors="ignore").to_csv(csv_path, index=False)

    return out_dir, stamp


# ---------------------------------------------------------------------------
# Form-field coercion
# ---------------------------------------------------------------------------

def coerce_str(v) -> str:
    """Coerce a Gradio form field to a stripped string.

    Gradio passes ``None`` for hidden components, and numeric components can
    feed values into textbox-adjacent paths. Anything not already a string is
    ``str()``-ified; the result is stripped. ``None`` becomes ``""``.
    """
    if v is None:
        return ""
    if not isinstance(v, str):
        v = str(v)
    return v.strip()


def clean_list(values: Iterable, *, dedupe: bool = False) -> list[str]:
    """Strip + drop-empties over ``values``; optionally case-insensitive dedupe.

    Turns the flattened textbox pool of a dynamic list (which includes blank
    hidden slots arriving as ``None``/``""``) back into a clean ``list[str]``.
    When ``dedupe`` is set, the first spelling of each case-folded value wins
    and later duplicates are dropped, preserving order.
    """
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = coerce_str(v)
        if not s:
            continue
        if dedupe:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Dynamic list-of-textboxes widget
# ---------------------------------------------------------------------------
#
# Gradio Blocks can't add/remove components at runtime, so an editable
# variable-length list is faked with a fixed pool of pre-allocated rows: the
# first ``len(defaults)`` are visible and prefilled, the rest hidden and
# revealed one at a time by "+ Add". Each list owns its own ``next_slot``
# cursor (a high-water mark for the Add button) so several lists can live on
# one page independently.
#
# This is the generic core shared by the scraper's query list and the RAG
# app's CV-theme columns. App-specific glue (reading values into a payload,
# hydrating from a profile, etc.) stays in the app modules.


@dataclass
class DynamicList:
    """Handles for one dynamic list-of-textboxes widget.

    ``boxes``/``rows``/``remove_btns`` are parallel lists of length
    ``pool_size``. ``next_slot`` is a ``gr.State`` holding the index of the
    next hidden row to reveal. ``add_btn``/``clear_btn`` are the list-level
    controls. Field types are left loose (``object``) so this module needn't
    import gradio at definition time.
    """
    title: str
    pool_size: int
    next_slot: object
    rows: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    remove_btns: list = field(default_factory=list)
    add_btn: object = None
    clear_btn: object = None


def build_dynamic_list(
    title: str,
    defaults: list[str],
    *,
    pool_size: int,
    placeholder: str = "",
    add_label: str = "+ Add",
    clear_label: str = "Clear all",
    min_visible: int = 1,
) -> "DynamicList":
    """Build a pool of pre-allocated rows for one editable list.

    Renders ``title`` as a column header, then ``pool_size`` rows. The first
    ``max(len(defaults), min_visible)`` rows are visible (defaults prefilled,
    any extra visible rows left blank); the rest are hidden and revealed by
    "+ Add". ``min_visible`` guarantees at least one editable row even when
    ``defaults`` is empty, which also avoids a column that renders as nothing
    but an Add button. Returns a :class:`DynamicList` of handles; call
    :func:`wire_dynamic_list` afterwards to attach the add/remove/clear
    behaviour. The caller owns the surrounding layout (e.g. wrapping several
    of these in a ``gr.Row`` of columns).
    """
    import gradio as gr

    if len(defaults) > pool_size:
        raise ValueError(
            f"{title!r}: {len(defaults)} defaults exceed pool_size {pool_size}"
        )

    if title:
        gr.Markdown(f"**{title}**")

    n_visible = min(max(len(defaults), min_visible), pool_size)
    next_slot = gr.State(n_visible)
    dl = DynamicList(title=title, pool_size=pool_size, next_slot=next_slot)

    for i in range(pool_size):
        prefill = defaults[i] if i < len(defaults) else ""
        with gr.Row(visible=(i < n_visible)) as row:
            tb = gr.Textbox(
                value=prefill,
                placeholder=placeholder,
                show_label=False,
                container=False,
                scale=20,
            )
            rm = gr.Button("✕", scale=1, variant="secondary", min_width=40)
        dl.rows.append(row)
        dl.boxes.append(tb)
        dl.remove_btns.append(rm)

    with gr.Row():
        dl.add_btn = gr.Button(add_label, variant="secondary")
        dl.clear_btn = gr.Button(clear_label, variant="secondary")

    return dl


def wire_dynamic_list(dl: "DynamicList") -> None:
    """Attach add/remove/clear ``.click`` handlers to a :class:`DynamicList`.

    Each list gets its own handlers closed over its own ``pool_size`` and
    component handles, so multiple lists on one page don't interfere. Removing
    a row hides+blanks it but does *not* rewind ``next_slot`` -- the cursor is
    a high-water mark for Add, not a live count; rows-with-empty-values are
    filtered out by :func:`clean_list` at read time anyway.
    """
    import gradio as gr

    pool = dl.pool_size

    def _add(n: int):
        # Reveal rows 0..n inclusive and advance the cursor. Revealing the
        # whole visible prefix (rather than just row n) is idempotent: a click
        # can never leave an earlier row hidden, even if a prior update was
        # dropped by the client. No-op past the end of the pool.
        if n >= pool:
            return [n, *[gr.update() for _ in range(pool)]]
        updates = [
            gr.update(visible=True) if i <= n else gr.update()
            for i in range(pool)
        ]
        return [n + 1, *updates]

    def _hide_and_clear():
        # Row + box to mutate are baked into `outputs=` at registration, so
        # this one function serves every remove button.
        return gr.update(visible=False), gr.update(value="")

    def _clear_all():
        return [gr.update(value="") for _ in range(pool)]

    dl.add_btn.click(_add, inputs=[dl.next_slot], outputs=[dl.next_slot, *dl.rows])

    for i, btn in enumerate(dl.remove_btns):
        btn.click(_hide_and_clear, inputs=None, outputs=[dl.rows[i], dl.boxes[i]])

    dl.clear_btn.click(_clear_all, inputs=None, outputs=dl.boxes)


def hydrate_dynamic_list(
    dl: "DynamicList", values: list[str], *, min_visible: int = 1
) -> list:
    """Return updates to (re)populate a list's pool from ``values``.

    Produces ``[next_slot_value, *row_updates, *box_updates]`` -- the first
    ``n`` rows visible (the first ``len(values)`` filled, any extra up to
    ``min_visible`` left blank), the rest hidden and blanked, and the cursor
    set to ``n``. Keeping ``min_visible`` rows means hydrating an empty
    profile still shows an editable row rather than collapsing the column to
    just its Add button. The output order matches
    ``[dl.next_slot, *dl.rows, *dl.boxes]``, which the caller must mirror in
    ``outputs=``. Extra values beyond ``pool_size`` are silently dropped.
    """
    import gradio as gr

    vals = list(values)[: dl.pool_size]
    n = min(max(len(vals), min_visible), dl.pool_size)
    row_updates = [
        gr.update(visible=(i < n)) for i in range(dl.pool_size)
    ]
    box_updates = [
        gr.update(value=(vals[i] if i < len(vals) else "")) for i in range(dl.pool_size)
    ]
    return [n, *row_updates, *box_updates]
