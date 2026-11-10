"""Prompt, cau tra loi co dinh va cac bang luat nam trong app/texts/, khong hard-code trong code. Sua file la sua hanh vi (build lai image).

- `*.txt`: mot van ban; dong trong o dau / cuoi bi bo.
- `*.json`: mot JSON object.
- `*.jsonl`: moi dong mot JSON object; dong trong va dong bat dau bang # bi bo qua. Luat regex co `pattern`, gia tri thay the (khoa do noi goi
  chi dinh), va `ignorecase` (tuy chon).
Sai o dau la nem ValueError ngay luc import, kem ten file (va so dong).
"""
import json
import re
from pathlib import Path

TEXTS_DIR = Path(__file__).parent / "texts"


def read_text(name):
    return (TEXTS_DIR / name).read_text(encoding="utf-8").strip("\n")


def read_json(name):
    try:
        return json.loads(read_text(name))
    except ValueError as e:
        raise ValueError(f"{name}: JSON khong hop le ({e})") from e


def read_rules(name, value_key):
    """[(regex da bien dich, row[value_key])] theo thu tu dong cua file `name`."""
    rules = []
    for number, line in enumerate((TEXTS_DIR / name).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
            rules.append((re.compile(row["pattern"], re.IGNORECASE if row.get("ignorecase") else 0), row[value_key]))
        except (ValueError, KeyError, TypeError, re.error) as e:
            raise ValueError(f"{name}:{number}: luat khong hop le ({type(e).__name__}: {e})") from e
    if not rules:
        raise ValueError(f"{name}: khong co luat nao")
    return rules
