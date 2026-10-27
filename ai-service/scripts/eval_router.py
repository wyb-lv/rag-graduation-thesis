"""Danh gia router LLM tren Gemini THAT (can GEMINI_API_KEY + ROUTER_MODEL/GEMINI_MODEL; ton quota). Khong chay trong CI.

    python -m scripts.eval_router [--limit N] [--rpm 15] [--timeout 15] [--no-clarify]

Moi cau/hoi thoai di qua router_llm.resolve + apply_policy, tuc la dung chinh chinh sach cua service (nguong, khong hoi hai lan, ...).
Cau don le duoc hoi mot minh (khong lich su, khong biet khoa); hoi thoai nhieu luot mo phong session: lich su, khoa va
previous_was_clarify duoc giu giua cac luot. Chi so chinh (muc tieu 0): ty le off_topic sai va clarify sai tren guard_legit.txt va tren cac
cau borderline/answerable. Khong con luat tu khoa de so sanh: baseline cu nam o git (app/guard.py, app/small_talk.py o commit 428c97c).
"""
import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

from app import config as cfg
from app.router import router_llm
from app.router.small_talk import Intent

DATA = Path(__file__).resolve().parent.parent / "tests" / "data"
_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
_HEADER = re.compile(r"\[[a-z_:]+\]")
STUB_ANSWER = "Theo quy định của trường, nội dung này nằm trong các văn bản liên quan."
NO_FACTS = {"khoa": None, "kho": None}


def sections(name):
    """{ten phan: [cau]} cua file corpus (dong `[ten]` mo phan moi, dong # la chu thich; \\n va \\uXXXX duoc giai ma)."""
    out, current = {}, None
    for line in (DATA / name).read_text(encoding="utf-8").splitlines():
        if _HEADER.fullmatch(line):  # "[INST] ... [/INST]" trong corpus tan cong khong phai tieu de phan
            current = out.setdefault(line.strip("[]"), [])
        elif line and not line.startswith("#"):
            text = _ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), line.replace("\\n", "\n"))
            (out.setdefault("", []) if current is None else current).append(text)
    return out


class Probe:
    """Dem so lan goi Gemini that, do do tre, va giu nhip theo RPM (limiter cua service bi nang len de khong can thiep)."""

    def __init__(self, rpm):
        self.calls, self.latencies, self.failures, self.n, self.gap = 0, [], 0, 0, 60 / rpm
        real = router_llm._complete

        async def counting(messages):
            self.calls += 1
            return await real(messages)

        router_llm._complete = counting

    async def decide(self, question, history=(), facts=NO_FACTS, previous=False):
        """(Routed, Action); Action la ket qua cua apply_policy, hoac RAG khi router loi (nhu service)."""
        router_llm._cache.clear()  # moi cau goi that, khong lan cache
        before = self.calls
        start = time.perf_counter()
        routed = await router_llm.resolve(question, list(history), facts, previous)
        self.latencies.append((time.perf_counter() - start) * 1000)
        self.n += 1
        if routed.decision is None:
            self.failures += 1
            action = router_llm.Action(Intent.RAG, query=question)
        else:
            action = router_llm.apply_policy(routed.decision, question, facts, previous)
        if self.calls > before:
            await asyncio.sleep(self.gap)
        return routed, action


def rate(count, total):
    return f"{count}/{total} = {count / total:.1%}" if total else "n/a"


def show(title, failed, limit=6):
    if failed:
        print(f"    {title}: " + " | ".join(q[:60].replace("\n", " ") for q in failed[:limit]) + (" ..." if len(failed) > limit else ""))


async def run_lines(probe, lines):
    return [(q, *(await probe.decide(q))) for q in lines]


async def main(args):
    probe = Probe(args.rpm)
    cut = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)
    attacks, legit = cut(sections("guard_attacks.txt")[""]), cut(sections("guard_legit.txt")[""])
    off = {k: cut(v) for k, v in sections("router_offtopic.txt").items()}
    clar = {k: cut(v) for k, v in sections("router_clarify.txt").items()}
    talk = {k: cut(v) for k, v in sections("router_smalltalk.txt").items()}
    print(f"model={cfg.ROUTER_MODEL} reasoning={cfg.ROUTER_REASONING_EFFORT} clarify_enabled={cfg.ROUTER_CLARIFY_ENABLED} rpm={args.rpm}")

    A, L = await run_lines(probe, attacks), await run_lines(probe, legit)
    OFF, BORDER = await run_lines(probe, off["off_topic"]), await run_lines(probe, off["borderline"])
    CT, CA, CM = [await run_lines(probe, clar[f"clarify:{m}"]) for m in ("topic", "audience", "time")]
    ANSW = await run_lines(probe, clar["answerable"])
    TALK = {k: await run_lines(probe, v) for k, v in talk.items()}

    is_ = lambda rows, intent: [r for r in rows if r[2].intent is intent]
    not_ = lambda rows, intent: [q for q, _, a in rows if a.intent is not intent]

    print("\n== Chong tan cong / chan nham (guard_attacks.txt, guard_legit.txt)")
    print("  attack recall (bi chan)            :", rate(len(is_(A, Intent.BLOCKED)), len(A)))
    show("lot luoi", [q for q, _, a in A if a.intent is not Intent.BLOCKED])
    print("  false block rate tren legit        :", rate(len(is_(L, Intent.BLOCKED)), len(L)))
    show("chan nham", [q for q, _, a in L if a.intent is Intent.BLOCKED])

    print("\n== Chi so chinh (muc tieu 0)")
    for label, rows in (("guard_legit", L), ("borderline", BORDER), ("answerable", ANSW)):
        print(f"  false off_topic / {label:11}: {rate(len(is_(rows, Intent.OFF_TOPIC)), len(rows))}")
        print(f"  false clarify   / {label:11}: {rate(len(is_(rows, Intent.CLARIFY)), len(rows))}")
        show("sai", [q for q, _, a in rows if a.intent in (Intent.OFF_TOPIC, Intent.CLARIFY)])

    print("\n== Off topic / clarify")
    print("  off_topic recall                   :", rate(len(is_(OFF, Intent.OFF_TOPIC)), len(OFF)))
    show("bo sot", not_(OFF, Intent.OFF_TOPIC))
    clar_rows = CT + CA + CM
    predicted = [r for r in L + BORDER + ANSW + OFF + clar_rows if r[2].intent is Intent.CLARIFY]
    right = [r for r in predicted if r in clar_rows]
    print("  clarify precision                  :", rate(len(right), len(predicted)))
    print("  clarify recall                     :", rate(len(is_(clar_rows, Intent.CLARIFY)), len(clar_rows)))
    for name, rows in (("topic", CT), ("audience", CA), ("time", CM)):
        ok = [r for r in rows if r[2].intent is Intent.CLARIFY and r[1].decision.missing == name]
        print(f"    dung `missing={name}`".ljust(38) + ":", rate(len(ok), len(rows)))

    print("\n== Accuracy y dinh (small talk + chao kem cau hoi + off_topic + clarify + legit)")
    expected = {"greeting": Intent.GREETING, "thanks": Intent.THANKS, "goodbye": Intent.GOODBYE, "capability": Intent.CAPABILITY,
                "meta": Intent.META, "rag": Intent.RAG}
    labelled = [(a.intent, expected[k]) for k, rows in TALK.items() for _, _, a in rows]
    labelled += [(a.intent, Intent.OFF_TOPIC) for _, _, a in OFF] + [(a.intent, Intent.RAG) for _, _, a in BORDER + ANSW + L]
    labelled += [(a.intent, Intent.CLARIFY) for _, _, a in clar_rows] + [(a.intent, Intent.BLOCKED) for _, _, a in A]
    print("  intent accuracy                    :", rate(sum(got is want for got, want in labelled), len(labelled)))
    for k, rows in TALK.items():
        show(f"sai [{k}]", [q for q, _, a in rows if a.intent is not expected[k]])

    print("\n== Hoi thoai nhieu luot (lich su + khoa + previous_was_clarify duoc giu giua cac luot)")
    kinds = {"clarify": [0, 0], "follow_up": [0, 0], "loop": [0, 0]}
    for dialog in json.loads((DATA / "router_dialogs.json").read_text(encoding="utf-8"))[:args.limit or None]:
        history, previous, khoa, ok = [], False, None, True
        for turn in dialog["turns"]:
            khoa = turn.get("khoa", khoa)
            routed, action = await probe.decide(turn["q"], history, {"khoa": khoa, "kho": None}, previous)
            want = Intent(turn["expect"])
            good = action.intent is want
            if good and turn.get("missing"):
                good = routed.decision is not None and routed.decision.missing == turn["missing"]
            if good and turn.get("query_contains"):
                good = all(w.lower() in (action.query or "").lower() for w in turn["query_contains"])
            if not good:
                ok = False
                print(f"    SAI {dialog['name']}: {turn['q']!r} -> {action.intent.value}"
                      + (f" query={action.query!r}" if action.query else ""))
            answer = action.answer or turn.get("answer", STUB_ANSWER)
            history += [("user", turn["q"]), ("assistant", answer)]
            previous = action.intent is Intent.CLARIFY
        kinds[dialog["kind"]][0] += ok
        kinds[dialog["kind"]][1] += 1
    print("  clarification completion success   :", rate(*kinds["clarify"]))
    print("  khong hoi hai lan lien tiep        :", rate(*kinds["loop"]))
    print("  follow up phu thuoc lich su (rag)  :", rate(*kinds["follow_up"]))

    lat = sorted(probe.latencies)
    print("\n== Van hanh")
    print("  fallback rate (router loi -> rag)  :", rate(probe.failures, probe.n))
    if lat:
        print(f"  latency p50 / p95 (ms)             : {statistics.median(lat):.0f} / {lat[min(len(lat) - 1, int(len(lat) * 0.95))]:.0f}")
    print("  Gemini calls                       :", probe.calls)
    print("  baseline tu khoa                   : da go bo (khong con de so sanh)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=0, help="chi lay N cau dau moi phan (chay thu nhanh, it quota)")
    parser.add_argument("--rpm", type=int, default=cfg.ROUTER_RPM, help="nhip goi Gemini moi phut (mac dinh ROUTER_RPM)")
    parser.add_argument("--timeout", type=float, default=15.0, help="timeout moi lan goi; cao hon ROUTER_TIMEOUT_S de do model, khong do mang")
    parser.add_argument("--no-clarify", action="store_true", help="ROUTER_CLARIFY_ENABLED=false: do tac dung cua clarify (ablation)")
    args = parser.parse_args()
    cfg.ROUTER_TIMEOUT_S = args.timeout
    cfg.ROUTER_RPM = 10_000  # nhip do Probe giu; limiter cua service khong duoc chen vao
    if args.no_clarify:
        cfg.ROUTER_CLARIFY_ENABLED = False
    if not cfg.GEMINI_API_KEY or not cfg.ROUTER_MODEL:
        sys.exit("Can GEMINI_API_KEY va ROUTER_MODEL (bien moi truong GEMINI_MODEL) trong ai-service/.env")
    asyncio.run(main(args))
