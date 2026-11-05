"""The Knowledge Pack: schema, scopes, documents, the CLI and kb_add."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from ledgerhound.kb import KBError, KnowledgePack, validate

RULE = {"type": "rule", "name": "high value new payee",
        "logic": "amount > 50000 and payee age < 1 day", "output_column": "rule_hit_17"}


@pytest.fixture
def team(tmp_path, monkeypatch):
    d = tmp_path / "team_kb"
    monkeypatch.setenv("LEDGERHOUND_TEAM_KB_DIR", str(d))
    return d


def test_builtin_pack_is_valid_and_covers_the_v1_typologies():
    pack = KnowledgePack(team=None)
    ids = {e["id"] for e in pack.entries(type="typology")}
    assert pack.load_errors == []
    assert ids == {"card_not_present", "account_takeover", "app_scam", "first_party_fraud",
                   "merchant_collusion", "mule_account"}
    pol = {e["label_source"]: e["maturity_days"] for e in pack.entries(type="label_policy")}
    assert pol == {"chargeback": 90, "complaint": 30, "any": 90}


@pytest.mark.parametrize("entry,needle", [
    ({"type": "rule", "name": "x"}, "missing required field 'logic'"),
    ({"type": "label_policy", "label_source": "cb", "maturity_days": "90"}, "must be int"),
    ({"type": "label_policy", "label_source": "cb", "maturity_days": True}, "must be an integer"),
    ({"type": "label_policy", "label_source": "cb", "maturity_days": 5000}, "between 0 and 1000"),
    ({"type": "dictionary", "column": "c", "meaning": "m", "role": "boss"}, "not a role"),
    ({"type": "typology", "name": "t", "description": "d", "roles_needed": ["payr"]}, "not a role"),
    ({"type": "rule", "name": "r", "logic": "l", "colour": "red"}, "unknown field 'colour'"),
    ({"type": "rule", "name": "r", "logic": "l", "status": "maybe"}, "status must be"),
    ({"type": "gossip"}, "type must be one of"),
    (["not", "a", "mapping"], "must be a mapping"),
])
def test_validation_refuses_with_the_fix(entry, needle):
    _, errs = validate(entry)
    assert errs and any(needle in e for e in errs), errs


def test_most_specific_scope_wins_and_latest_line_wins(tmp_path, team):
    pack = KnowledgePack()
    pack.add({"type": "label_policy", "label_source": "chargeback", "maturity_days": 120},
             scope="team", added_by="t")
    assert pack.get("label_policy", "chargeback").data["maturity_days"] == 120
    _, replaced = pack.add({"type": "label_policy", "label_source": "Chargeback",
                            "maturity_days": 60}, scope="project", added_by="p")
    assert replaced
    e = pack.get("label_policy", "chargeback")
    assert (e.scope, e.data["maturity_days"]) == ("project", 60)
    pack.add({"type": "label_policy", "label_source": "chargeback", "maturity_days": 75},
             scope="project", added_by="p")
    assert KnowledgePack().get("label_policy", "chargeback").data["maturity_days"] == 75
    # append-only: every version is still on disk
    lines = (tmp_path / ".ledgerhound" / "kb" / "entries.jsonl").read_text().splitlines()
    assert [json.loads(x)["maturity_days"] for x in lines] == [60, 75]


def test_builtin_is_read_only_and_team_needs_a_folder(monkeypatch):
    monkeypatch.delenv("LEDGERHOUND_TEAM_KB_DIR", raising=False)
    pack = KnowledgePack()
    with pytest.raises(KBError, match="read-only"):
        pack.add(RULE, scope="builtin")
    with pytest.raises(KBError, match="no team folder"):
        pack.add(RULE, scope="team")


def test_documents_are_stored_with_their_text(tmp_path):
    doc = tmp_path / "sop.md"
    doc.write_text("# SOP\nEscalate first payments over 1 lakh to a new payee.\n")
    pack = KnowledgePack()
    e = pack.add_document(doc, source="ops SOP v3", added_by="t")
    assert e["source"] == "ops SOP v3" and e["chars"] > 10
    assert "new payee" in pack.doc_text(pack.get("document", e["id"]))
    with pytest.raises(KBError, match="no file"):
        pack.add_document(tmp_path / "missing.md")
    bad = tmp_path / "evil.sh"
    bad.write_text("rm -rf /")
    with pytest.raises(KBError, match="documents must be"):
        pack.add_document(bad)


def test_pdf_without_the_extra_says_how_to_install(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def no_pypdf(name, *a, **k):
        if name == "pypdf":
            raise ImportError("no pypdf")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_pypdf)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(KBError, match=r"ledgerhound\[docs\]"):
        KnowledgePack().add_document(pdf)


def test_corrupt_lines_are_skipped_and_reported(tmp_path):
    pack = KnowledgePack()
    pack.add(RULE, added_by="t")
    f = tmp_path / ".ledgerhound" / "kb" / "entries.jsonl"
    f.write_text(f.read_text() + "{not json\n" + json.dumps({"type": "rule", "name": "n"}) + "\n")
    fresh = KnowledgePack()
    assert fresh.get("rule", "high_value_new_payee") is not None
    assert len(fresh.load_errors) == 2


def test_cli_add_list_validate(tmp_path):
    from ledgerhound.kb.cli import kb_app
    run = CliRunner().invoke
    f = tmp_path / "e.yaml"
    f.write_text("- type: rule\n  name: r1\n  logic: amount > 10\n"
                 "- type: dictionary\n  column: c\n  meaning: m\n  role: bogus\n")
    assert run(kb_app, ["validate", str(f)]).exit_code == 1
    r = run(kb_app, ["add", str(f)])
    assert r.exit_code == 1 and "added rule r1" in r.output
    r = run(kb_app, ["list", "--type", "rule"])
    assert "[project] rule/r1" in r.output
    assert run(kb_app, ["validate"]).exit_code == 0
