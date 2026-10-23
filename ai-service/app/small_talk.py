"""Dinh tuyen y dinh: chao / cam on / tam biet / hoi ve tro ly duoc tra loi co dinh (khong truy van, khong goi LLM).

Chi khop khi ca tin nhan ngan (<= MAX_WORDS tu) va moi menh de (tach bang dau cau) khop tron mot cum da biet, kem tu dem
nhu "ban", "thay", "nhe". Gap gi khong chac thi chon `rag` (hoi them mot cau con hon tra loi sai mot cau chao).
Khop tren ban fold (chu thuong, bo dau) nen "Xin chào", "xin chao", "XIN CHAO 😊" nhu nhau.
"""
import re
from enum import StrEnum

from . import config as cfg
from .guard import fold

MAX_WORDS = 8


class Intent(StrEnum):
    RAG = "rag"
    GREETING = "greeting"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    CAPABILITY = "capability"
    META = "meta"  # hoi ve model / prompt / cau hinh noi bo
    BLOCKED = "blocked"  # chi dung lam nhan phan hoi khi guard chan; route_intent khong bao gio tra ve


_FILLER = (
    "ban|bot|chatbot|ad|admin|oi|a|nhe|nha|nhieu|ne|nhi|thay|co|anh|chi|em|minh|toi|tui|moi nguoi|please|pls|plz|there|"
    "team|everyone|guys|friend|da|vay|day|ha|di|rat|lam|that|so much|a lot|cho toi|cho em|cho minh|duoc khong|"
    "phai khong|vui long|lam on|xin|tro ly|hsu|hoa sen"
)
_AI = "chatgpt|gpt|openai|claude|gemini|llama|qwen|deepseek|copilot|bard"
_CORES = {
    Intent.GREETING: (
        r"(?:xin )?chao(?: buoi (?:sang|trua|chieu|toi))?|hello|helo|hi|hey|heyy|hii|hiya|halo|alo|yo|hola|greetings|chao mung"
        r"|good (?:morning|afternoon|evening|day)|buoi (?:sang|trua|chieu|toi) tot lanh"
    ),
    Intent.THANKS: (
        r"(?:(?:xin|da hieu|ok|oke|okay|vang) )?(?:cam on|cam ta|biet on|thanks|thank you|thank u|thankyou|thx|tks|ty|cheers)"
        r"(?: (?:nhieu|rat nhieu|so much|a lot|very much|vi tat ca))*"
    ),
    Intent.GOODBYE: (
        r"tam biet|bye|bye bye|byebye|goodbye|good bye|see you|see ya|gap lai sau|hen gap lai|chao tam biet|xin tam biet"
        r"|chuc ban (?:mot )?ngay tot lanh"
    ),
    Intent.CAPABILITY: (
        r"ban la ai|ban la gi|who are you|what are you|(?:hay )?gioi thieu(?: ve)?(?: ban than| ban| chatbot| bot)?"
        r"|ban lam duoc (?:nhung )?gi|ban co the lam (?:duoc )?(?:nhung )?gi|ban giup duoc (?:nhung )?gi"
        r"|ban co the giup (?:duoc )?(?:toi )?(?:nhung )?gi|ban ho tro (?:duoc )?(?:nhung )?gi|ban biet lam gi|ban biet nhung gi"
        r"|ban dung de lam gi|ban de lam gi|ban co (?:nhung )?(?:chuc nang|cong dung|tinh nang) gi"
        r"|(?:cong dung|chuc nang|muc dich|tac dung|vai tro) cua (?:ban|chatbot|bot|tro ly)(?: la gi)?"
        r"|what can you do|what do you do|what are you for|how can you help(?: me)?|how do you work|can you help me"
        r"|help(?: me)?|tro giup|giup do|toi can giup do"
    ),
    Intent.META: (
        r"ban (?:dang )?(?:dung|su dung|xai|chay (?:bang|tren)|dua tren|duoc xay dung (?:bang|tren)) "
        r"(?:model|mo hinh|llm|ai|gpt|" + _AI + r"|cong nghe|ngon ngu)(?: (?:nao|gi))?"
        r"|ban la (?:model|mo hinh|llm) (?:nao|gi)"
        r"|(?:model|mo hinh|llm|prompt|system prompt|cau hinh|huong dan noi bo) cua ban(?: la gi| nhu the nao| ra sao)?"
        r"|ban (?:co phai )?(?:la )?(?:" + _AI + r")(?: phai khong| khong)?"
        r"|(?:ai|cong ty nao|ben nao|hang nao) (?:da )?(?:tao ra|lam ra|phat trien|xay dung|huan luyen|lap trinh|viet ra) ban"
        r"|ban (?:duoc|do) (?:ai|cong ty nao|ben nao|hang nao) (?:tao ra|tao|lam ra|phat trien|xay dung|huan luyen|lap trinh)"
        r"|ban (?:duoc )?(?:huan luyen|lap trinh|cau hinh|tao ra|xay dung|hoat dong) (?:nhu the nao|the nao|ra sao|nhu nao|bang gi|boi ai)"
        r"|(?:what|which) (?:model|llm|ai|gpt) (?:are you|do you use|are you using|is this|powers you|r u)"
        r"|(?:are you|r u) (?:" + _AI + r"|an ai|a bot|a robot|human|real)"
        r"|who (?:made|created|built|trained|developed|programmed) you"
        r"|what is your (?:system )?(?:model|config|configuration)"
    ),
}
# Uu tien khi nhieu menh de ("cam on, tam biet", "chao ban, ban dung model nao"): meta > capability > goodbye > thanks > greeting
_ORDER = (Intent.META, Intent.CAPABILITY, Intent.GOODBYE, Intent.THANKS, Intent.GREETING)
_FULL = {  # tu dem, mot cum cua y dinh, roi tu dem hoac lap lai cum ("alo alo", "cam on cam on")
    intent: re.compile(rf"(?:(?:{_FILLER}) )*(?:{core})(?: (?:{_FILLER}|{core}))*") for intent, core in _CORES.items()
}
_CLAUSES = re.compile(r"[,.;!?\n]+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def route_intent(text):
    """Intent cua mot tin nhan: fixed intent chi khi tin nhan ngan va moi menh de khop tron; con lai la `rag`."""
    if len(re.findall(r"\w+", text)) > MAX_WORDS:
        return Intent.RAG
    clauses = [c for c in (_NON_ALNUM.sub(" ", fold(c)).strip() for c in _CLAUSES.split(text)) if c]
    found = [next((i for i in _ORDER if _FULL[i].fullmatch(c)), None) for c in clauses]
    if not found or None in found:
        return Intent.RAG
    return min(found, key=_ORDER.index)


_OPENER = re.compile(
    r"(?:(?:xin )?chao|hello|helo|hi|hey|alo|halo|good (?:morning|afternoon|evening)|(?:chuc )?buoi (?:sang|trua|chieu|toi) tot lanh)"
    r"(?: (?:ban|thay|co|anh|chi|em|ad|admin|bot|chatbot|oi|a|nhe|nha|moi nguoi|tro ly|team|there)){0,2}(?= |$)"
)


def _rests(text):
    """Phan con lai sau loi chao, tu bo nhieu tu nhat (chao + goi "thay"/"ban") den chi bo tu chao; giu nguyen dau / viet hoa."""
    tokens = list(re.finditer(r"\w+", text))
    m = _OPENER.match(" ".join(fold(t.group()) for t in tokens))
    for k in range(len(m.group().split()), 0, -1) if m else ():
        rest = text[tokens[k - 1].end():].lstrip(" \t\n,.;:!?-–—…")
        if rest:
            yield rest


def strip_opener(text):
    """Bo loi chao dau cau ("chào thầy, cho em hỏi ..." -> "cho em hỏi ..."); khong co loi chao hoac khong con gi thi tra ve text goc."""
    return next(_rests(text), text)


def route(text):
    """(intent, cau hoi dung cho RAG). "Chao thay, cho em hoi ve hoc bong" -> (RAG, "cho em hoi ve hoc bong").
    Bo loi chao co the lo ra mot y dinh co dinh: "hello cam on" -> (THANKS, "cam on"); "hello ban lam duoc gi" -> CAPABILITY."""
    intent = route_intent(text)
    if intent is not Intent.RAG:
        return intent, text
    rests = list(_rests(text))
    for rest in rests:
        if (found := route_intent(rest)) is not Intent.RAG:
            return found, rest
    return Intent.RAG, rests[0] if rests else text


def reply_for(intent):
    return getattr(cfg, f"GUARD_REPLY_{intent.name}")
