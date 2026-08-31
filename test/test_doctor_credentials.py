"""The doctor Credentials section — the self-service answer to "AWS is unavailable".

Advisory by construction: an unconfigured AWS profile is not a Kiro Crew fault, so
every case here also asserts that ``issues`` stays empty. A regression that made
this section blocking would turn ``doctor`` red on every host that does not use
AWS.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiro_crew import cli_doctor


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(cli_doctor, "_credential_vendor_line", lambda: "")
    # Default to "no AWS CLI to ask": a case that is not ABOUT the probes must
    # not spawn one, and `None` is the shape that means "could not ask".
    monkeypatch.setattr(cli_doctor, "_aws_profile_names", lambda: None)
    monkeypatch.setattr(cli_doctor, "_aws_auto_refreshes", lambda: False)
    return tmp_path


class TestCredentialsSection:
    def test_no_aws_config_is_reported_without_failing(self, fake_home, capsys):
        issues: list[str] = []
        cli_doctor._doctor_credentials(issues)
        out = capsys.readouterr().out
        assert "Credentials" in out
        assert "no ~/.aws config" in out
        assert issues == []

    def test_profiles_are_listed(self, fake_home, monkeypatch, capsys):
        aws = fake_home / ".aws"
        aws.mkdir()
        (aws / "config").write_text("[default]\n")
        monkeypatch.setattr(cli_doctor, "_aws_profile_names", lambda: ["default", "build"])
        issues: list[str] = []
        cli_doctor._doctor_credentials(issues)
        out = capsys.readouterr().out
        assert "default" in out
        assert "build" in out
        assert issues == []

    def test_profile_set_is_unknown_without_the_cli(self, fake_home, capsys):
        """``None`` means "could not ask", which is not "there are none".

        The files exist and the config is not ours to parse, so the honest report
        names the gap instead of implying an empty profile set.
        """
        aws = fake_home / ".aws"
        aws.mkdir()
        (aws / "config").write_text("[default]\n")
        cli_doctor._doctor_credentials([])
        assert "install the AWS CLI to list profiles" in capsys.readouterr().out

    def test_credential_process_is_called_out(self, fake_home, monkeypatch, capsys):
        aws = fake_home / ".aws"
        aws.mkdir()
        (aws / "config").write_text("[profile p]\n")
        monkeypatch.setattr(cli_doctor, "_aws_auto_refreshes", lambda: True)
        issues: list[str] = []
        cli_doctor._doctor_credentials(issues)
        out = capsys.readouterr().out
        assert "credential_process configured" in out
        assert issues == []

    def test_absent_credential_process_is_reported(self, fake_home, capsys):
        aws = fake_home / ".aws"
        aws.mkdir()
        (aws / "config").write_text("[profile p]\n")
        issues: list[str] = []
        cli_doctor._doctor_credentials(issues)
        assert "no credential_process" in capsys.readouterr().out
        assert issues == []

    def test_credentials_file_alone_still_reports_a_profile(self, fake_home, monkeypatch, capsys):
        aws = fake_home / ".aws"
        aws.mkdir()
        (aws / "credentials").write_text("[default]\n")
        monkeypatch.setattr(cli_doctor, "_aws_profile_names", lambda: [])
        issues: list[str] = []
        cli_doctor._doctor_credentials(issues)
        assert "default" in capsys.readouterr().out
        assert issues == []

    def test_nothing_under_dot_aws_is_opened(self, fake_home, monkeypatch, capsys):
        """The section must not READ a single byte out of ``~/.aws``.

        That directory is fenced from the agent by the sensitive-path floor, and
        ``kirocrew doctor`` is reachable from a tool call — so parsing the config
        here would hand back through a diagnostic exactly what the floor refuses
        directly. Existence probes are fine; opening is not. Asserted on the OPEN
        rather than on the printed output, because "no secret appeared in stdout"
        would still pass while the bytes were being read.
        """
        aws = fake_home / ".aws"
        aws.mkdir()
        (aws / "config").write_text("[profile p]\ncredential_process = /usr/bin/vend\n")
        (aws / "credentials").write_text("[p]\naws_secret_access_key = SUPERSECRETVALUE\n")
        opened: list[str] = []
        real_read_text = Path.read_text
        real_open = Path.open

        def _record_read_text(self, *a, **kw):
            opened.append(str(self))
            return real_read_text(self, *a, **kw)

        def _record_open(self, *a, **kw):
            opened.append(str(self))
            return real_open(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", _record_read_text, raising=True)
        monkeypatch.setattr(Path, "open", _record_open, raising=True)
        cli_doctor._doctor_credentials([])
        assert not [p for p in opened if ".aws" in p], f"the section opened {opened}"

    def test_no_secret_value_is_printed(self, fake_home, capsys):
        """Belt to the brace above: even the profile NAME channel carries no value."""
        aws = fake_home / ".aws"
        aws.mkdir()
        (aws / "config").write_text("[profile p]\nregion = us-west-2\n")
        (aws / "credentials").write_text("[p]\naws_secret_access_key = SUPERSECRETVALUE\n")
        cli_doctor._doctor_credentials([])
        out = capsys.readouterr().out
        assert "SUPERSECRETVALUE" not in out
        assert "aws_secret_access_key" not in out

    def test_the_misdiagnosis_note_is_always_present(self, fake_home, capsys):
        """This line is the point of the section — it must not be conditional."""
        cli_doctor._doctor_credentials([])
        out = capsys.readouterr().out
        assert "cannot READ credential files" in out
        assert "blocked-commands.md" in out

    def test_vendor_line_is_shown_when_the_edition_has_one(self, fake_home, monkeypatch, capsys):
        monkeypatch.setattr(
            cli_doctor, "_credential_vendor_line", lambda: "may vend credentials (creds-agent)"
        )
        issues: list[str] = []
        cli_doctor._doctor_credentials(issues)
        out = capsys.readouterr().out
        assert "vending MCP" in out
        assert "creds-agent" in out
        assert issues == []


class _Proc:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


@pytest.fixture
def spawns(monkeypatch):
    """Record every argv the probes would run, without running one."""
    recorded: list[list[str]] = []
    monkeypatch.setattr(cli_doctor.shutil, "which", lambda name: f"/usr/bin/{name}")

    def _install(result):
        def _run(argv, **kw):
            recorded.append(list(argv))
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(cli_doctor.subprocess, "run", _run)
        return recorded

    return _install


class TestProfileProbeUsesTheSanctionedPath:
    def test_argv_is_fixed_and_interpolates_nothing(self, spawns):
        recorded = spawns(_Proc(stdout="default\nbuild\n"))
        assert cli_doctor._aws_profile_names() == ["default", "build"]
        assert recorded == [["/usr/bin/aws", "configure", "list-profiles"]]

    def test_duplicates_and_blank_lines_are_dropped(self, spawns):
        spawns(_Proc(stdout="a\n\na\nb\n  \n"))
        assert cli_doctor._aws_profile_names() == ["a", "b"]

    def test_no_cli_means_could_not_ask(self, monkeypatch):
        monkeypatch.setattr(cli_doctor.shutil, "which", lambda name: None)
        assert cli_doctor._aws_profile_names() is None

    @pytest.mark.parametrize(
        "result",
        [_Proc(returncode=1, stdout="boom"), OSError("no exec"), Exception("timeout")],
    )
    def test_a_failed_probe_is_could_not_ask_not_empty(self, spawns, result):
        """A failure must not read as "this host has no profiles"."""
        spawns(result)
        assert cli_doctor._aws_profile_names() is None


class TestAutoRefreshProbe:
    def test_a_configured_process_is_reported(self, spawns):
        recorded = spawns(_Proc(stdout="/usr/bin/vend\n"))
        assert cli_doctor._aws_auto_refreshes() is True
        assert recorded == [["/usr/bin/aws", "configure", "get", "credential_process"]]

    @pytest.mark.parametrize(
        "result",
        [_Proc(stdout="  \n"), _Proc(returncode=1, stdout="/usr/bin/vend"), OSError("no exec")],
    )
    def test_anything_short_of_a_value_is_false(self, spawns, result):
        spawns(result)
        assert cli_doctor._aws_auto_refreshes() is False

    def test_no_cli_is_false(self, monkeypatch):
        monkeypatch.setattr(cli_doctor.shutil, "which", lambda name: None)
        assert cli_doctor._aws_auto_refreshes() is False


class TestVendorLineIsFailSoft:
    def test_public_edition_yields_no_line(self):
        """The public default reports available() False, so nothing is probed."""
        assert cli_doctor._credential_vendor_line() == ""

    def test_a_lookup_error_degrades_to_no_line(self, monkeypatch):
        import kiro_crew.platform.context as ctx_mod

        def _boom(fn, **kw):
            raise RuntimeError("composition exploded")

        monkeypatch.setattr(ctx_mod, "safe_context_call", _boom, raising=True)
        assert cli_doctor._credential_vendor_line() == ""
