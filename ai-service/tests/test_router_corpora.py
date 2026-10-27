"""Corpus + script danh gia router (scripts/eval_router.py). Offline: chi kiem tra du lieu va bo doc, khong goi Gemini."""
import json

from scripts import eval_router as ev

DIALOGS = json.loads((ev.DATA / "router_dialogs.json").read_text(encoding="utf-8"))


def test_corpora_meet_the_minimum_sizes():
    off, clar = ev.sections("router_offtopic.txt"), ev.sections("router_clarify.txt")
    assert len(off["off_topic"]) >= 40 and len(off["borderline"]) >= 15
    assert sum(len(v) for k, v in clar.items() if k.startswith("clarify:")) >= 20 and len(clar["answerable"]) >= 15
    assert set(clar) == {"clarify:topic", "clarify:audience", "clarify:time", "answerable"}
    assert len(ev.sections("guard_attacks.txt")[""]) >= 40 and len(ev.sections("guard_legit.txt")[""]) >= 40
    assert set(ev.sections("router_smalltalk.txt")) == {"greeting", "thanks", "goodbye", "capability", "meta", "rag"}


def test_both_diacritic_and_plain_ascii_spellings_and_english_are_covered():
    lines = [l for v in ev.sections("router_offtopic.txt").values() for l in v] + [l for v in ev.sections("router_clarify.txt").values() for l in v]
    assert any(not l.isascii() for l in lines) and any(l.isascii() and "dieu kien" in l.lower() for l in lines)
    assert any(l.isascii() and l.startswith(("What", "write", "Can", "Recommend", "Translate")) for l in lines)


def test_attack_lines_that_start_with_a_bracket_are_not_mistaken_for_section_headers():
    attacks = ev.sections("guard_attacks.txt")
    assert list(attacks) == [""] and any(l.startswith("[INST]") for l in attacks[""])


def test_no_line_is_shared_between_a_must_flag_list_and_a_must_stay_rag_list():
    off, clar = ev.sections("router_offtopic.txt"), ev.sections("router_clarify.txt")
    flagged = set(off["off_topic"]) | {l for k, v in clar.items() if k.startswith("clarify:") for l in v}
    stay_rag = set(off["borderline"]) | set(clar["answerable"]) | set(ev.sections("guard_legit.txt")[""])
    assert not flagged & stay_rag


def test_dialog_fixtures_cover_completion_loop_and_history_dependent_follow_ups():
    assert {d["kind"] for d in DIALOGS} == {"clarify", "loop", "follow_up"}
    for d in DIALOGS:
        assert d["turns"] and all(t["expect"] in {"rag", "clarify", "greeting"} for t in d["turns"])
    assert any(t.get("missing") == "audience" and "khoa" in d["turns"][1] for d in DIALOGS for t in d["turns"][:1])
    assert any(len(d["turns"]) >= 3 for d in DIALOGS)
