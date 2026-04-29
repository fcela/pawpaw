from __future__ import annotations

import pytest

from pawpaw import cli


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "spec_path" in out or "spec" in out


def test_cli_rejects_missing_spec(tmp_path):
    code = cli.main([str(tmp_path / "nope.txt"), "-o", str(tmp_path / "out.paw")])
    assert code != 0
