"""``kirocrew.session.duration`` + ``kirocrew.session.started``.

How long a session lives, and why it ended. Together they answer questions the
existing session instruments cannot: ``kirocrew.session.startup.duration``
measures the cold start of the agent PROCESS, and
``kirocrew.session.idle_expired`` counts one specific teardown cause, but
nothing recorded how long a session actually lasted or how its lifetimes split
across the ways a session can end.

**Why the reason is passed in, never derived.** There is no single teardown
function. ``session_lifecycle.SessionLifecycleService`` exposes six distinct end
paths -- ``reset``, ``remove``, ``remove_if_unclaimed``, ``destroy``,
``discard_conversation`` and ``close_all`` -- and each pops the registry itself
rather than delegating to a shared funnel. Two more paths pop it WITHOUT going
through any of those: ``retire_kiro_identity_sessions`` (an identity change
retires every idle process on the old account) and
``CompactionCoordinator._recycle_held`` (context overflow replaces the provider
in place). So each path states its own reason, the same contract
``metrics/turns.py`` holds its callers to.

``reset`` is the widest of them: the idle sweep and a slot reset both reach
teardown through it, so ``end_reason=reset`` is a path label, not a cause. That
is deliberate -- the finer causes behind it are already counted separately
(``kirocrew.session.idle_expired``, ``kirocrew.watchdog.recovery.outcome``), and
a metric must not be the reason a lifecycle signature every surface calls grows
a parameter. The identity retire and the compaction recycle are NOT inside it and
carry their own labels, because a metric that reported them as ``reset`` would be
claiming a teardown route they never take.

**Why a breadcrumb on disk.** A start time held only in memory dies with the
process, which is exactly the population most worth measuring: a session that
ends because the gateway crashed never runs any teardown path, so it would
contribute no sample at all and the histogram would describe only orderly
shutdowns. Each start therefore drops one small JSON file under
``<data home>/metrics/open-sessions/``; a clean end reads it, emits, and unlinks
it. Whatever is still there at the next boot belongs to a session that never
ended cleanly, so :func:`backfill_crashed_sessions` emits it as
``end_reason=crashed`` and unlinks it.

**Why not the transcript's existing close marker.** A transcript's metadata line
already carries ``created_at``, and the dashboard tab-close path stamps
``closed`` / ``closed_at`` beside it. Neither substitutes for the crumb. That
stamp is written by the dashboard close path ALONE, so a channel, cron, subagent
or task-runner session never gets one; and its ABSENCE is ambiguous by
construction -- a transcript with no ``closed`` is equally a crashed session, a
live-but-idle one, and one that was never a dashboard tab. There is no positive
crash flag on disk today, which is what the crumb supplies, for every surface.
Boot has no session sweep to piggyback on either: the restore path is seed-driven
from ``open_slots.json`` and never walks the transcript directory.

**Why the clean path reads the crumb rather than the session in hand.** The
popped registry entry does carry an epoch ``created_at``, so a clean end could
compute its own duration without touching disk. It deliberately does not, for two
reasons. The clean and crashed paths must consume the SAME record or they
double-count: a crumb left behind by a clean end is indistinguishable at the next
boot from one left by a crash, and would be emitted a second time as
``crashed``. And that field is RESET when a provider is recycled in place, so it
measures the provider's age rather than the session's residency in the registry.

That unlink is what makes the accounting exact. A crumb is consumed at most
once, so the six teardown paths cannot double-count a session between them (the
idle sweep's ``reset`` is a real instance of this: the sweep and the path it
calls both sit on the same session), and a backfilled session cannot be counted
again on the boot after that.

**Where a crashed session's END time comes from.** The crumb is written once and
never rewritten, so its own mtime is its start. The session's transcript
(``<data home>/sessions/<stem>.jsonl``) is appended to as the conversation runs,
so its mtime is the last moment the session was observably alive -- the honest
maximum available after the fact. A session with no transcript on disk (a
subagent leaves only a replay log) yields no end time, so it is unlinked without
an emit rather than recorded as a plausible-looking zero: an absent sample reads
as "no data", a zero renders as a real 0ms lifetime.

Telemetry must never break the instrumented path, so every function here
swallows its failures after a debug log, and every attribute value is a
constant from the closed enums below.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

#: Session lifetime, in ms. Registered in ``metrics/provider.py``'s bucket table
#: under its own boundary family: this is the only kirocrew histogram whose range
#: is minutes to days, so it shares no family with the request/turn instruments.
SESSION_DURATION_METRIC = "kirocrew.session.duration"

#: One increment per session admitted to the registry -- the denominator the
#: duration histogram's population is a subset of.
SESSION_STARTED_COUNTER = "kirocrew.session.started"

# ---------------------------------------------------------------------------
# end_reason -- closed enum, one member per teardown path
# ---------------------------------------------------------------------------

#: ``SessionLifecycleService.reset`` -- the widest path (idle sweep, slot reset,
#: watchdog recycle). A compaction recycle is NOT here: it pops the registry
#: itself and reports :data:`END_REASON_RECYCLED`. See the module docstring.
END_REASON_RESET = "reset"
#: ``remove`` -- shut down, session-map entry deliberately preserved.
END_REASON_REMOVED = "removed"
#: ``remove_if_unclaimed`` -- a speculative session whose first turn never came.
END_REASON_UNCLAIMED = "unclaimed"
#: ``destroy`` -- session and its persistence entry both gone.
END_REASON_DESTROYED = "destroyed"
#: ``discard_conversation`` -- native conversation dropped, linkage kept.
END_REASON_DISCARDED = "discarded"
#: ``close_all`` -- gateway shutdown drained the registry.
END_REASON_SHUTDOWN = "shutdown"
#: ``retire_kiro_identity_sessions`` -- the account behind the session changed,
#: so every idle process on the old identity is torn down. Its own label rather
#: than folded into ``reset``: it pops the registry directly and never calls
#: ``reset``, and an identity change is a different event from an idle sweep.
END_REASON_RETIRED = "retired"
#: ``CompactionCoordinator._recycle_held`` -- context overflow replaced the
#: provider in place. Also pops directly rather than routing through ``reset``.
END_REASON_RECYCLED = "recycled"
#: Backfilled at boot from a crumb no teardown path ever consumed.
END_REASON_CRASHED = "crashed"

#: Every value ``end_reason`` can carry. The drift gate in
#: ``test/metrics/test_session_duration.py`` harvests this set, so a new path
#: cannot start emitting a label the tests do not know about.
END_REASONS = frozenset(
    {
        END_REASON_RESET,
        END_REASON_REMOVED,
        END_REASON_UNCLAIMED,
        END_REASON_DESTROYED,
        END_REASON_DISCARDED,
        END_REASON_SHUTDOWN,
        END_REASON_RETIRED,
        END_REASON_RECYCLED,
        END_REASON_CRASHED,
    }
)

# A runaway crumb directory must not turn boot into a stat storm, so the backfill
# emits at most this many samples per boot. Everything it walks is unlinked
# either way, so a directory that somehow grew past the cap is drained rather
# than left to be re-walked at every boot from then on.
_MAX_BACKFILL_EMITS = 2000

_CRUMB_DIR_NAME = "open-sessions"

# Start time per LIVE session key -- the authority for the normal path, so an end
# never has to read the disk and can therefore run in the same tick as the
# registry removal it reports. The crumb on disk exists only to survive process
# death. Last write wins: a key re-entering the registry is a new session whose
# lifetime must be its own (see record_session_started).
#
# **This table is also the crumb's GENERATION TOKEN, and the only one.** The
# crumb write is handed to a worker thread, so the writer can land after its
# session already ended, or after a SUCCESSOR registered under the same key --
# either way a crumb describing a session nobody will consume, which the next
# boot reports as a crash that never happened. Rather than track ended keys
# separately, the writer re-reads this table and writes only while its own
# ``started_at`` is still the installed generation: an ended key has been popped,
# and a superseded key holds the successor's newer value, so both cases refuse
# with one comparison. There is no second set to keep in step with this one.
#
# Bounded because not every registry removal records an end, so a stale entry is
# possible; the cap discards the oldest half rather than growing for the process
# lifetime, which costs samples and never memory. An evicted key's in-flight
# writer therefore refuses -- the same "costs samples, never memory" trade, and
# it takes _MAX_LIVE_SESSIONS concurrently live sessions to reach.
_MAX_LIVE_SESSIONS = 4096
_live_lock = threading.Lock()
_live_starts: dict[str, float] = {}


def _is_current_generation(session_key: str, started_at: float) -> bool:
    """True while *started_at* is still the generation installed for *session_key*.

    False once the key has been popped by an end, or overwritten by a successor's
    start. Both are reasons an in-flight crumb write must not land.
    """
    with _live_lock:
        return _live_starts.get(session_key) == started_at


def _crumb_dir() -> Path:
    """``<data home>/metrics/open-sessions`` -- created on demand."""
    from kiro_crew.config.paths import config_dir

    return config_dir() / "metrics" / _CRUMB_DIR_NAME


def _crumb_path(session_key: str) -> Path:
    """Path of *session_key*'s crumb, named by digest rather than by the key.

    Session keys carry channel ids and slot names and are not constrained to
    filesystem-safe characters, so the key is never used as a filename. The key
    itself is stored INSIDE the file, which is what lets the backfill find the
    session's transcript; it is not new exposure, since ``session_map.json``
    already holds every session key in the same data home.
    """
    digest = hashlib.sha256(session_key.encode("utf-8", "surrogatepass")).hexdigest()
    return _crumb_dir() / f"{digest[:32]}.json"


def _session_source(session_key: str) -> str:
    """Bounded label for which surface owns *session_key*.

    ``telemetry_channel_of`` exists for exactly this question and never returns
    the key itself, so cardinality is bounded whatever the caller passes. Same
    derivation ``metrics/turns.py`` uses, so the two instruments group by the
    same label.
    """
    from kiro_crew.messaging.link import telemetry_channel_of

    return telemetry_channel_of(session_key)


def record_session_started(session_key: str) -> None:
    """Record one session start; never raises.

    The start time goes into an in-memory table keyed by session key, and the
    crumb on disk is written off-thread. The table is what the normal path reads;
    the crumb exists ONLY so a start can outlive its process (see
    :func:`backfill_crashed_sessions`).

    **The table entry is overwritten, and so is the crumb.** A key re-entering
    the registry is a NEW session, and its lifetime must be measured from ITS
    start, not from a predecessor's -- registry removal has no single choke point
    (twelve call sites remove from it, and not all of them record an end), so a
    stale entry is a real possibility and last-writer-wins is what keeps a
    successor's lifetime its own. The cost of a removal that records nothing is
    then one MISSING sample, never a lifetime spanning two sessions.

    Safe to call while the session registry lock is held: the table write and the
    counter are in-memory, and only the crumb touches disk, on the shared
    maintenance pool. That pool is submitted to DIRECTLY rather than through
    ``loop.run_in_executor``, whose future is bound to the running loop -- a
    fire-and-forget write outliving its loop raises "Event loop is closed" when
    the result is set, which is what Windows CI caught.
    """
    if not session_key:
        return
    started_at = time.time()
    # Installing the generation is itself what lifts any predecessor's claim on
    # this key: an earlier session's in-flight crumb writer compares against this
    # table and refuses once it no longer matches.
    with _live_lock:
        _live_starts[session_key] = started_at
        if len(_live_starts) > _MAX_LIVE_SESSIONS:
            # A registry this large is not real; drop the oldest half rather than
            # growing for the process lifetime. Costs samples, never memory.
            oldest = sorted(_live_starts.items(), key=lambda kv: kv[1])
            for stale_key, _stale_at in oldest[: len(oldest) // 2]:
                _live_starts.pop(stale_key, None)
    try:
        from kiro_crew.metrics.events import emit_counter

        emit_counter(SESSION_STARTED_COUNTER, {"session_source": _session_source(session_key)})
    except Exception:
        logger.debug("session started counter failed", exc_info=True)
    _submit(_write_crumb, session_key, started_at)


def _submit(fn, *args) -> None:
    """Run *fn* off-thread on the maintenance pool, or inline if it is gone."""
    try:
        from kiro_crew.executors import maintenance_executor

        maintenance_executor().submit(fn, *args)
    except Exception:
        # No pool (interpreter shutdown, a test that closed it): the work is a
        # single small file operation, so do it inline rather than drop it.
        logger.debug("session crumb handoff failed; running inline", exc_info=True)
        try:
            fn(*args)
        except Exception:
            logger.debug("session crumb inline fallback failed", exc_info=True)


def _write_crumb(session_key: str, started_at: float) -> None:
    """Persist *session_key*'s open-session crumb; never raises.

    Overwrites any existing crumb, for the reason
    :func:`record_session_started` gives: the newest start is the one a crash
    should be measured from. Refuses unless *started_at* is still the generation
    installed for this key, because this runs on a worker thread and two things
    can overtake it: the session can reach teardown first (the key is popped), or
    a SUCCESSOR can register under the same key (the key holds a newer value).
    Writing in either case would leave a crumb nothing consumes, which the next
    boot reads as a crash that never happened -- or, worse, would measure the
    successor's crash from this session's start.
    """
    if not _is_current_generation(session_key, started_at):
        return
    try:
        from kiro_crew.atomic_write import atomic_write

        path = _crumb_path(session_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            path,
            json.dumps({"key": session_key, "started_at": started_at}, ensure_ascii=True),
        )
        # Re-checked after the write: the end, or a successor's start, could have
        # landed while it was in flight, and the crumb must not outlive the
        # session it describes.
        if not _is_current_generation(session_key, started_at):
            _unlink(path)
    except Exception:
        logger.debug("session start crumb write failed", exc_info=True)


def record_session_ended(session_key: str, *, end_reason: str) -> None:
    """Emit *session_key*'s lifetime; never raises.

    **Non-blocking, so it belongs in the same tick as the registry removal it
    reports.** The start time is popped from the in-memory table, the histogram
    is recorded in memory, and the only disk work is a single ``unlink``. That
    is what lets every teardown path call this immediately after its
    ``_sessions.pop`` and BEFORE its first ``await``. The earlier version read
    the crumb off disk here, which forced the call to the end of teardown -- and
    a replacement session registering under the same key during those awaits then
    had its crumb consumed by its predecessor's teardown.

    Popping the table entry is also the double-count defence: the teardown paths
    are not mutually exclusive (the idle sweep calls ``reset``), so whichever
    reaches the session first is the one that records it, and a second call finds
    nothing.
    **The crumb unlink runs INLINE, and inside the generation lock.** It is one
    ``unlink`` syscall, so it costs this tick almost nothing, and handing it to
    the maintenance pool instead was wrong twice over. A queued unlink can land
    AFTER a successor registered under the same key and wrote its own crumb,
    deleting a live session's crumb and losing its later crash. And
    ``shutdown_maintenance_executor`` drains with ``cancel_futures=True``, so at
    ``close_all`` -- exactly when that pool is flooded with teardown work -- the
    unlink is cancelled outright, leaving behind the crumb of a session that
    ended cleanly for the next boot to back-fill as ``crashed``: orderly shutdown
    would inflate the very population this instrument exists to measure. Running
    it here, under the same lock a successor's start takes to install its
    generation, means no successor can interleave between the pop and the unlink
    and nothing can be cancelled.
    """
    if not session_key or end_reason not in END_REASONS:
        return
    # Built before the lock so only the syscall is held inside it.
    crumb = _crumb_path(session_key)
    with _live_lock:
        started_at = _live_starts.pop(session_key, None)
        # Unlinked whether or not a duration was emitted: the session is over, so
        # its crumb must not survive for the next boot to read as a crash.
        _unlink(crumb)
    if started_at is None:
        return
    try:
        _emit_duration(session_key, time.time() - started_at, end_reason)
    except Exception:
        logger.debug("session end emit failed", exc_info=True)


def backfill_crashed_sessions(started_before: float | None = None) -> int:
    """Emit ``end_reason=crashed`` for crumbs no teardown path consumed.

    *started_before* is an epoch cutoff: a crumb whose ``started_at`` is at or
    after it is left untouched. Callers pass their own process start time, which
    is what makes this safe to run OFF the boot path -- without the cutoff the
    scan had to complete before this process opened its first session, or it
    would back-fill a crumb it had just written as though a previous run had
    crashed. With it, ordering is irrelevant: only crumbs that predate this
    process can ever be claimed. ``None`` disables the cutoff (tests).

    Returns the number of samples emitted (0 on any failure); never raises.
    """
    emitted = 0
    try:
        crumb_dir = _crumb_dir()
        if not crumb_dir.is_dir():
            return 0
        for path in sorted(crumb_dir.glob("*.json")):
            key, started_at = _read_crumb(path)
            if started_at is not None and started_before is not None:
                if started_at >= started_before:
                    # This process's own crumb: still live, not a casualty.
                    continue
            _unlink(path)
            if not key or started_at is None or emitted >= _MAX_BACKFILL_EMITS:
                continue
            ended_at = _last_activity(key)
            if ended_at is None:
                continue
            if _emit_duration(key, ended_at - started_at, END_REASON_CRASHED):
                emitted += 1
    except Exception:
        logger.debug("crashed-session backfill failed", exc_info=True)
    return emitted


def _emit_duration(session_key: str, seconds: float, end_reason: str) -> bool:
    """Record one histogram sample. Returns whether a sample was emitted.

    A non-positive lifetime is skipped rather than recorded, for the reason
    ``metrics/turns.py`` gives: an absent sample reads as "no data" on the
    Telemetry page, while a recorded 0 renders as a plausible 0ms lifetime.
    """
    if seconds <= 0:
        return False
    try:
        from kiro_crew.metrics.provider import get_recorder

        get_recorder().histogram(
            SESSION_DURATION_METRIC,
            seconds * 1000.0,
            unit="ms",
            attrs={"end_reason": end_reason, "session_source": _session_source(session_key)},
            description="Session lifetime (ms), by how the session ended.",
        )
        return True
    except Exception:
        logger.debug("session duration emit failed", exc_info=True)
        return False


def _read_crumb(path: Path) -> tuple[str, float | None]:
    """Return ``(session_key, started_at)`` from *path*, or ``("", None)``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "", None
    if not isinstance(payload, dict):
        return "", None
    key = payload.get("key")
    started = payload.get("started_at")
    if not isinstance(key, str) or not isinstance(started, (int, float)):
        return "", None
    return key, float(started)


def _last_activity(session_key: str) -> float | None:
    """Mtime of *session_key*'s transcript -- when it was last observably alive.

    ``None`` when the session has no transcript on disk, which is normal rather
    than an error: a subagent run leaves only a replay log. The caller skips the
    emit in that case (see the module docstring).
    """
    try:
        from kiro_crew.config.paths import config_dir
        from kiro_crew.history import SESSIONS_DIR_NAME, transcript_stem

        path = config_dir() / SESSIONS_DIR_NAME / f"{transcript_stem(session_key)}.jsonl"
        return path.stat().st_mtime
    except Exception:
        return None


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("session crumb unlink failed for %s", path.name, exc_info=True)
