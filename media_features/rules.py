"""First-line rules: number_in_title and question_in_title are defined by these rules (VLM answers are used only
to check agreement).
  number_in_title   = the first line contains digits 0-9 (numbers written as words do not count; empty line -> 0);
  question_in_title = the first line contains "?" or starts (after emoji/punctuation) with a Russian question word:
                      who, what, how, why, what for, where, when, how many, which (any form), "is it possible",
                      "do you know" (empty line -> 0). The Russian words are written as unicode escapes below.
"""
import re

QWORDS = re.compile(
    r"^(\u043a\u0442\u043e|\u0447\u0442\u043e|\u043a\u0430\u043a|\u043f\u043e\u0447\u0435\u043c\u0443|\u0437\u0430\u0447\u0435\u043c|\u0433\u0434\u0435|\u043a\u043e\u0433\u0434\u0430|\u0441\u043a\u043e\u043b\u044c\u043a\u043e|\u043a\u0430\u043a(\u043e\u0439|\u0430\u044f|\u043e\u0435|\u0438\u0435|\u043e\u0433\u043e|\u043e\u043c\u0443|\u0438\u043c|\u0438\u0445|\u0443\u044e|\u0438\u043c\u0438)|\u043c\u043e\u0436\u043d\u043e\s+\u043b\u0438|\u0437\u043d\u0430\u0435\u0442\u0435\s+\u043b\u0438)\b",
    re.I)
LEAD = re.compile(r"^[^\w]+", re.U)


def has_digits(s) -> bool:
    return bool(re.search(r"[0-9]", s or ""))


def is_question(s) -> bool:
    s = s or ""
    if "?" in s:
        return True
    return bool(QWORDS.match(LEAD.sub("", s).replace("\u0451", "\u0435")))
