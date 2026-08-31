"""``kirocrew.session.duration`` + ``kirocrew.session.started``.

Drives the REAL production helpers in ``metrics/sessions.py`` with a patched
recorder and a redirected data home, so the crumb lifecycle, the exactly-once
consumption, the end_reason enum and the crashed back-fill all live in
production code -- a change there fails these tests instead of passing green.
"""

import json
import time
from unittest.mock import patch

import pytest

from kiro_crew.metrics import sessions as sess


class _CapturingRecorder:
    """Stand-in recorder that captures histogram() and counter() calls."""

    def __init__(self) -> None:
        self.hist: list = []
        self.counters: list = []

    def histogram(self, name, value, *, unit="ms", attrs=None, **kwargs) -> None:
        self.hist.append({"name": name, "value": value, "unit": unit, "attrs": dict(attrs or {})})

    def counter(self, name, value=1, *, attrs=None, **kwargs) -> None:
        self.counters.append({"name": name, "value": value, "attrs": dict(attrs or {})})


#: The real submitter, captured before the autouse fixture below replaces it, so
#: the one test that must exercise the POOL itself can put it back.
_REAL_SUBMIT = sess._submit


@pytest.fixture(autouse=True)
def clean_live_registry():
    """The live-start table is process-global; leaking it across tests is ordering bugs."""
    with sess._live_lock:
        sess._live_starts.clear()
    yield
    with sess._live_lock:
        sess._live_starts.clear()


@pytest.fixture(autouse=True)
def crumbs_are_synchronous(monkeypatch):
    """Run crumb work inline instead of on the shared maintenance pool.

    Review finding: the ``home`` fixture redirects ``config_dir`` for the duration
    of ONE test, but a crumb write handed to the pool outlives it -- a busy pool
    lets the worker land after teardown has restored the real data home, so the
    test writes a crumb into the developer's own install. Running it inline keeps
    every crumb inside that test's ``tmp_path``, and makes crumb state
    deterministic to assert instead of something to poll.
    """
    monkeypatch.setattr(sess, "_submit", lambda fn, *args: fn(*args))


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point the crumb directory and transcript store at a temp data home."""
    monkeypatch.setattr("kiro_crew.config.paths.config_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def rec():
    r = _CapturingRecorder()
    with patch("kiro_crew.metrics.provider.get_recorder", return_value=r):
        yield r


def _crumbs(home):
    d = home / "metrics" / "open-sessions"
    return sorted(d.glob("*.json")) if d.is_dir() else []


def _write_crumb(home, key, started_at):
    """Leave behind the crumb a CRASHED process would leave.

    Goes through the production writer, which only writes while the key's
    generation is installed, then drops the generation again -- the crash itself,
    which takes the whole in-memory table with it and is precisely why the crumb
    on disk is the only surviving evidence.
    """
    with sess._live_lock:
        sess._live_starts[key] = started_at
    sess._write_crumb(key, started_at)
    with sess._live_lock:
        sess._live_starts.pop(key, None)
    assert _crumbs(home), "helper must have produced a crumb"


def _duration_calls(rec):
    return [c for c in rec.hist if c["name"] == "kirocrew.session.duration"]


class TestStart:
    def test_start_counts_and_leaves_a_crumb(self, home, rec):
        sess.record_session_started("dashboard:chat-1")
        names = [c["name"] for c in rec.counters]
        assert "kirocrew.session.started" in names
        assert len(_crumbs(home)) == 1

    def test_start_labels_the_surface(self, home, rec):
        sess.record_session_started("dashboard:chat-1")
        started = [c for c in rec.counters if c["name"] == "kirocrew.session.started"][-1]
        assert started["attrs"]["session_source"] == "dashboard"

    def test_crumb_is_named_by_digest_not_by_the_key(self, home, rec):
        # A session key can carry path separators and channel punctuation; a
        # key-named file would escape the directory or fail to create at all.
        sess.record_session_started("slack:C123/../../etc/passwd")
        crumbs = _crumbs(home)
        assert len(crumbs) == 1
        assert "passwd" not in crumbs[0].name

    def test_a_second_start_overwrites_the_first(self, home, rec):
        """Deliberate reversal, found in review round 2.

        Registry removal has no single choke point and not every remover records
        an end, so a stale entry is possible. A key re-entering the registry is a
        NEW session whose lifetime must be its own -- keeping the predecessor's
        start would report a lifetime spanning two sessions.
        """
        sess.record_session_started("dashboard:chat-1")
        first = sess._live_starts["dashboard:chat-1"]
        time.sleep(0.01)
        sess.record_session_started("dashboard:chat-1")
        assert sess._live_starts["dashboard:chat-1"] > first

    def test_empty_key_is_a_no_op(self, home, rec):
        sess.record_session_started("")
        assert not rec.counters
        assert not _crumbs(home)


class TestEnd:
    @staticmethod
    def _start(key, seconds_ago):
        """Register a live session the way production does, aged by seconds."""
        sess.record_session_started(key)
        with sess._live_lock:
            sess._live_starts[key] = time.time() - seconds_ago

    def test_end_emits_the_lifetime_and_consumes_the_crumb(self, home, rec):
        self._start("dashboard:chat-1", 60)
        assert _crumbs(home), "the start must have left a crumb to consume"
        sess.record_session_ended("dashboard:chat-1", end_reason=sess.END_REASON_RESET)
        calls = _duration_calls(rec)
        assert len(calls) == 1
        assert calls[0]["unit"] == "ms"
        assert 55_000 < calls[0]["value"] < 70_000
        assert calls[0]["attrs"] == {"end_reason": "reset", "session_source": "dashboard"}
        assert _crumbs(home) == [], "the end must consume the crumb"

    def test_the_end_record_touches_no_disk(self):
        """It runs in the same tick as the registry pop, so it cannot block.

        Review round 2: reading the crumb here forced the call to the end of
        teardown, and a replacement session registering under the same key during
        those awaits had its start consumed by its predecessor.
        """
        import inspect

        body = inspect.getsource(sess.record_session_ended)
        assert "_live_starts.pop" in body
        assert "_read_crumb" not in body
        assert "read_text" not in body

    def test_a_second_end_emits_nothing(self, home, rec):
        """The teardown paths overlap -- the idle sweep calls reset."""
        self._start("dashboard:chat-1", 60)
        sess.record_session_ended("dashboard:chat-1", end_reason=sess.END_REASON_RESET)
        sess.record_session_ended("dashboard:chat-1", end_reason=sess.END_REASON_SHUTDOWN)
        assert len(_duration_calls(rec)) == 1

    def test_end_without_a_crumb_emits_nothing(self, home, rec):
        sess.record_session_ended("dashboard:never-started", end_reason=sess.END_REASON_RESET)
        assert not _duration_calls(rec)

    def test_unknown_end_reason_is_refused(self, home, rec):
        """An unbounded label would mint a series; the enum is the gate."""
        _write_crumb(home, "dashboard:chat-1", time.time() - 60)
        sess.record_session_ended("dashboard:chat-1", end_reason="whatever-i-like")
        assert not _duration_calls(rec)
        assert _crumbs(home), "a refused reason must not consume the crumb"

    def test_corrupt_crumb_is_consumed_without_emitting(self, home, rec):
        _write_crumb(home, "dashboard:chat-1", time.time() - 60)
        _crumbs(home)[0].write_text("not json at all")
        sess.record_session_ended("dashboard:chat-1", end_reason=sess.END_REASON_RESET)
        assert not _duration_calls(rec)
        assert not _crumbs(home), "a corrupt crumb must not be re-walked every boot"

    def test_non_positive_lifetime_is_skipped(self, home, rec):
        # A LIVE start in the future, so the end actually reaches the emit and is
        # rejected there. Planting only a crumb would return early at the pop and
        # pass without ever exercising the guard.
        self._start("dashboard:chat-1", -3600)
        sess.record_session_ended("dashboard:chat-1", end_reason=sess.END_REASON_RESET)
        assert not _duration_calls(rec)


class TestCrashedBackfill:
    def _transcript(self, home, key, mtime):
        from kiro_crew.history import SESSIONS_DIR_NAME, transcript_stem

        d = home / SESSIONS_DIR_NAME
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{transcript_stem(key)}.jsonl"
        path.write_text("{}\n")
        import os

        os.utime(path, (mtime, mtime))
        return path

    def test_leftover_crumb_becomes_a_crashed_sample(self, home, rec):
        started = time.time() - 7200
        _write_crumb(home, "dashboard:chat-1", started)
        self._transcript(home, "dashboard:chat-1", started + 3600)
        assert sess.backfill_crashed_sessions() == 1
        calls = _duration_calls(rec)
        assert len(calls) == 1
        assert calls[0]["attrs"]["end_reason"] == "crashed"
        assert 3_500_000 < calls[0]["value"] < 3_700_000

    def test_backfill_consumes_the_crumb_so_it_cannot_recur(self, home, rec):
        started = time.time() - 7200
        _write_crumb(home, "dashboard:chat-1", started)
        self._transcript(home, "dashboard:chat-1", started + 3600)
        sess.backfill_crashed_sessions()
        assert not _crumbs(home)
        assert sess.backfill_crashed_sessions() == 0
        assert len(_duration_calls(rec)) == 1

    def test_no_transcript_means_no_sample_but_the_crumb_still_goes(self, home, rec):
        _write_crumb(home, "subagent:run-1", time.time() - 7200)
        assert sess.backfill_crashed_sessions() == 0
        assert not _duration_calls(rec)
        assert not _crumbs(home)

    def test_missing_directory_is_not_an_error(self, home, rec):
        assert sess.backfill_crashed_sessions() == 0

    def test_a_cleanly_ended_session_is_never_backfilled(self, home, rec):
        started = time.time() - 7200
        sess.record_session_started("dashboard:chat-1")
        with sess._live_lock:
            sess._live_starts["dashboard:chat-1"] = started
        assert _crumbs(home), "the start must have left a crumb to consume"
        self._transcript(home, "dashboard:chat-1", started + 3600)
        sess.record_session_ended("dashboard:chat-1", end_reason=sess.END_REASON_SHUTDOWN)
        assert _crumbs(home) == [], "the end must consume the crumb"
        assert sess.backfill_crashed_sessions() == 0
        calls = _duration_calls(rec)
        assert len(calls) == 1
        assert calls[0]["attrs"]["end_reason"] == "shutdown"


class TestContract:
    def test_every_end_reason_constant_is_in_the_enum(self):
        """A new path must not emit a label the tests do not know about."""
        declared = {
            value
            for name, value in vars(sess).items()
            if name.startswith("END_REASON_") and isinstance(value, str)
        }
        assert declared == set(sess.END_REASONS)

    def test_the_lifecycle_module_uses_only_enum_members(self):
        """The teardown paths import their labels rather than spelling them."""
        from kiro_crew import session_lifecycle

        for name in (
            "END_REASON_RESET",
            "END_REASON_REMOVED",
            "END_REASON_UNCLAIMED",
            "END_REASON_DESTROYED",
            "END_REASON_DISCARDED",
            "END_REASON_SHUTDOWN",
        ):
            assert getattr(session_lifecycle, name) in sess.END_REASONS

    def test_the_duration_histogram_has_registered_bounds(self):
        from kiro_crew.metrics.provider import _HISTOGRAM_BUCKETS_MS

        bounds = _HISTOGRAM_BUCKETS_MS[sess.SESSION_DURATION_METRIC]
        # Minutes to days: a week-long dashboard tab must not land in +Inf, and
        # a seconds-long unclaimed session must not collapse onto bound one.
        assert bounds[0] <= 1000
        assert bounds[-1] >= 7 * 24 * 60 * 60 * 1000


class TestReviewFixes:
    """Regressions found in review -- each of these was a real defect."""

    def test_a_writer_landing_after_the_end_leaves_no_crumb(self, home, rec):
        """The deferred write could manufacture a crash that never happened."""
        sess.record_session_ended("dashboard:chat-1", end_reason=sess.END_REASON_RESET)
        # The worker thread runs LATE, after teardown already consumed nothing.
        sess._write_crumb("dashboard:chat-1", time.time() - 60)
        assert not _crumbs(home), "a crumb written after the end would backfill as crashed"
        assert sess.backfill_crashed_sessions() == 0

    def test_a_later_session_under_the_same_key_still_gets_a_crumb(self, home, rec):
        """An ended key must not be suppressed for the rest of the process."""
        sess.record_session_ended("dashboard:chat-1", end_reason=sess.END_REASON_RESET)
        sess.record_session_started("dashboard:chat-1")
        assert len(_crumbs(home)) == 1

    def test_a_superseded_writer_does_not_overwrite_the_successors_crumb(self, home, rec):
        """Review round 4: the crumb needs a generation, not just an end flag.

        A predecessor's write can still be in flight when a SUCCESSOR registers
        under the same key. An end flag says nothing about that case -- the
        successor's start lifts it -- so the stale writer would land and the
        successor's crash would be measured from the predecessor's start.
        """
        sess.record_session_started("dashboard:chat-1")
        current = sess._live_starts["dashboard:chat-1"]
        # The predecessor's worker thread runs LATE, with its own older start.
        sess._write_crumb("dashboard:chat-1", current - 3600)
        crumbs = _crumbs(home)
        assert len(crumbs) == 1
        assert json.loads(crumbs[0].read_text())["started_at"] == current

    def test_the_end_unlinks_inline_so_the_unlink_cannot_be_cancelled(self):
        """Review round 4: a pooled unlink was wrong in both directions.

        Queued, it can land after a successor wrote its own crumb and delete it,
        losing that session's later crash. And ``shutdown_maintenance_executor``
        drains with ``cancel_futures=True``, so at ``close_all`` -- when the pool
        is flooded with teardown work -- the unlink is dropped entirely, leaving
        a cleanly ended session's crumb for the next boot to call ``crashed``.
        """
        import inspect

        body = inspect.getsource(sess.record_session_ended)
        assert "_submit(" not in body, "the unlink must not be handed to the pool"
        assert "_unlink(crumb)" in body

    def test_the_backfill_never_claims_a_crumb_from_this_process(self, home, rec):
        """The cutoff is what lets the scan run off the boot path."""
        cutoff = time.time()
        _write_crumb(home, "dashboard:chat-1", cutoff + 5)
        assert sess.backfill_crashed_sessions(cutoff) == 0
        assert _crumbs(home), "a live crumb must survive the scan"

    def test_the_backfill_still_claims_an_older_crumb_under_a_cutoff(self, home, rec):
        started = time.time() - 7200
        _write_crumb(home, "dashboard:chat-1", started)
        TestCrashedBackfill()._transcript(home, "dashboard:chat-1", started + 3600)
        assert sess.backfill_crashed_sessions(time.time()) == 1
        assert _duration_calls(rec)[0]["attrs"]["end_reason"] == "crashed"

    def test_retire_and_recycle_have_their_own_reasons(self):
        """They pop the registry directly and never route through reset."""
        assert sess.END_REASON_RETIRED in sess.END_REASONS
        assert sess.END_REASON_RECYCLED in sess.END_REASONS
        assert sess.END_REASON_RETIRED != sess.END_REASON_RESET
        assert sess.END_REASON_RECYCLED != sess.END_REASON_RESET

    def test_the_identity_retire_path_records_an_end(self):
        from kiro_crew import session_lifecycle

        found = TestTeardownPathsAreWired._reasons_recorded_by("retire_kiro_identity_sessions")
        assert found == ["END_REASON_RETIRED"]
        assert session_lifecycle.END_REASON_RETIRED in sess.END_REASONS

    def test_the_compaction_recycle_path_records_an_end(self):
        import ast
        import inspect

        from kiro_crew import session_compaction

        tree = ast.parse(inspect.getsource(session_compaction))
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_recycle_held":
                for call in ast.walk(node):
                    if not isinstance(call, ast.Call):
                        continue
                    fn = call.func
                    name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                    if name == "record_session_ended":
                        found.append(
                            [
                                k.value.id
                                for k in call.keywords
                                if k.arg == "end_reason" and isinstance(k.value, ast.Name)
                            ]
                        )
        # Exactly one: the branch that owns the pop. The other branch's key
        # already holds a SUCCESSOR whose crumb must not be consumed here.
        assert found == [["END_REASON_RECYCLED"]]

    def test_the_boot_hook_no_longer_blocks_readiness(self):
        from pathlib import Path

        src = (Path(sess.__file__).resolve().parent.parent / "slack" / "gateway.py").read_text(
            encoding="utf-8"
        )
        i = src.index("_backfill_unclean_session_telemetry")
        window = src[i : i + 900]
        assert "asyncio.to_thread" in window, "the scan must not run on the loop"
        assert "create_task" in window, "the scan must not be awaited before readiness"
        assert "_background_tasks" in window, "the task must be tracked, not fire-and-forget"

    def test_the_crumb_write_is_not_bound_to_an_event_loop(self):
        """Windows CI failed with 'Event loop is closed' on the first version.

        ``loop.run_in_executor`` hands back a future owned by the running loop,
        so a fire-and-forget write that outlives its loop raises when the result
        is set. Submitting to the shared pool directly has no loop affinity.
        """
        import inspect

        body = inspect.getsource(sess.record_session_started)
        assert "run_in_executor(" not in body
        # _REAL_SUBMIT, not sess._submit: the autouse fixture replaces the latter
        # with an inline double, whose source says nothing about the pool.
        assert "maintenance_executor" in inspect.getsource(_REAL_SUBMIT)

    def test_a_start_with_no_running_loop_still_writes_its_crumb(self, home, rec, monkeypatch):
        # The only test that opts OUT of the inline-crumb fixture: the point here
        # is that the REAL pool has no loop affinity. Waiting on the outcome is
        # also what keeps it honest about the fixture's reason for existing --
        # the worker has demonstrably finished before this test's home goes away.
        monkeypatch.setattr(sess, "_submit", _REAL_SUBMIT)
        sess.record_session_started("cron:job-1")
        # The pool is real, so wait on the OUTCOME rather than on which path
        # produced it -- inline fallback and pooled write are both acceptable.
        for _ in range(100):
            if _crumbs(home):
                break
            time.sleep(0.02)
        assert _crumbs(home), "a session start must leave a crumb with no loop running"


class TestTeardownPathsAreWired:
    """Each teardown path must record, with a reason of its own.

    A source-level gate rather than a driven one: these six methods are the
    lifecycle service's own registry pops, and standing each of them up needs a
    stub owner, provider, executor and platform layer -- which would pin the
    stubs, not the wiring. The behaviour they delegate to is driven directly
    above; what this holds is that they still delegate at all, and that no two
    paths report the same reason (which would silently merge two populations).
    """

    EXPECTED = {
        "reset": "END_REASON_RESET",
        "remove": "END_REASON_REMOVED",
        "remove_if_unclaimed": "END_REASON_UNCLAIMED",
        "destroy": "END_REASON_DESTROYED",
        "discard_conversation": "END_REASON_DISCARDED",
        "close_all": "END_REASON_SHUTDOWN",
    }

    @staticmethod
    def _reasons_recorded_by(method_name):
        import ast
        import inspect

        from kiro_crew import session_lifecycle

        tree = ast.parse(inspect.getsource(session_lifecycle))
        found: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef) or node.name != method_name:
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if name != "record_session_ended":
                    continue
                for kw in call.keywords:
                    if kw.arg == "end_reason" and isinstance(kw.value, ast.Name):
                        found.append(kw.value.id)
        return found

    @pytest.mark.parametrize("method,const", sorted(EXPECTED.items()))
    def test_the_path_records_with_its_own_reason(self, method, const):
        assert self._reasons_recorded_by(method) == [const]

    def test_no_two_paths_share_a_reason(self):
        reasons = [c for m in self.EXPECTED for c in self._reasons_recorded_by(m)]
        assert len(reasons) == len(set(reasons))

    def test_the_boot_backfill_is_wired_into_gateway_startup(self):
        from pathlib import Path

        gateway = Path(sess.__file__).resolve().parent.parent / "slack" / "gateway.py"
        src = gateway.read_text(encoding="utf-8")
        assert "backfill_crashed_sessions" in src
        # Ordering against the orphan sweep is deliberately NOT asserted any
        # more: review showed an inline pre-ready scan delays readiness in
        # proportion to accumulated crumbs, so it moved to a worker thread. What
        # replaces the ordering requirement is the process-start cutoff, which
        # TestReviewFixes pins directly -- the scan can no longer mistake a
        # session THIS process opened for a casualty of the last one.
        assert "_telemetry_backfill_cutoff" in src

    def test_the_registry_insertions_record_a_start(self):
        from pathlib import Path

        root = Path(sess.__file__).resolve().parent.parent
        for rel in ("session_allocation.py", "session_background.py"):
            src = (root / rel).read_text(encoding="utf-8")
            assert "record_session_started" in src, rel
