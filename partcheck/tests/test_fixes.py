"""Regression tests for the bugs fixed in the review (each one failed before the fix).     Run:  python tests/test_fixes.py"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triage_core as tc
import triage_ui as ui

FAILS, PASSES = [], 0


def check(name, cond):
    global PASSES
    PASSES += cond
    if not cond:
        FAILS.append(name)
    print(("PASS  " if cond else "FAIL  ") + name)


dmg = lambda t: bool(tc.rule_flags([t]).loc[0, "damaged_defective"])
check("'missing mounting hardware' is a damage/defect phrase (the rule used to need the exact words 'missing hardware')", dmg("It came with missing mounting hardware"))
check("'missing all the screws' too", dmg("missing all the screws"))
names = lambda t: [p["name"] for p in tc.mentioned_parts(t)]
check("'went nuts' is not the part 'Screws, bolts & nuts'", names("I went nuts over how good this is") == [])
check("'lightweight' is not a light bulb", names("Nice and lightweight") == [])
check("a real mention is still found", names("The lug nuts were stripped") == ["Lug nuts / wheel locks"])

photo = {"status": "part", "families": {"interior": 0.97, "wheels": 0.02}, "family": "interior", "shows_name": "floor mats"}
two = [{"key": "tires", "name": "Tyres"}, {"key": "floor_mats", "name": "Floor mats"}]
check("a review naming two parts matches a picture of the SECOND one (it used to look only at the first)", ui.compare(two, photo)["state"] == "match")
check("a picture that matches none of the named parts is still a mismatch", ui.compare([two[0]], photo)["state"] == "mismatch")


class Up(io.BytesIO):
    def __init__(self, b, name):
        super().__init__(b)
        self.name = name


rd = lambda b, n: ui.read_table("", Up(b, n))[0]
check("a Windows-1252 CSV is read", list(rd("review,rating\ncafé broke,1\n".encode("cp1252"), "a.csv").columns) == ["review", "rating"])
check("a semicolon CSV is read", rd(b"review;rating\nbroke;1\n", "b.csv").shape == (1, 2))
check("a CSV with a BOM has a clean first column name", list(rd("\ufeffreview,rating\nok,5\n".encode("utf-8"), "c.csv").columns)[0] == "review")
try:
    rd(b"", "e.csv"); ok = False
except ui.InputError:
    ok = True
check("an empty file gives a readable error", ok)
print(f"\n{PASSES} passed, {len(FAILS)} failed")
sys.exit(1 if FAILS else 0)
