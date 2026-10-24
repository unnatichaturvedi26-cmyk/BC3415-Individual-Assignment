"""Shared building blocks for the review-triage project (reused by every notebook and the final app).

v1 = first version (kept so we can measure the improvement honestly).
v2 = broadened using (a) phrasings seen in the failures on labelled Set 1 and (b) general wording.
"""
import re
import numpy as np
import pandas as pd

ISSUES = ["damaged_defective", "wrong_item", "not_as_described", "size_fit", "delivery_shipping", "customer_service"]
CODES = dict(zip(ISSUES, ["D", "W", "A", "S", "L", "C"]))

# ---------------- v1 keyword rules (transparent and fast, but blind to context) ----------------
PATTERNS_V1 = {
    "damaged_defective": r"\b(?:broke|broken|breaks|cracked|cracks?|damaged|defective|defect|faulty|malfunction\w*|stopped working|stops working|doesn['’]?t work|does not work|didn['’]?t work|did not work|not working|dead on arrival|leak(?:s|ed|ing)?|torn|ripped|scratched|dented?|bent|smashed|shattered|fell apart|falling apart|came apart|missing (?:parts?|pieces?|screws?)|chipped)\b",
    "wrong_item": r"\b(?:wrong (?:item|size|colou?r|product|model|one|hand|part|version|flavou?r)|not what i ordered|not the item i ordered|different (?:item|product|model)|incorrect (?:item|size|colou?r|product)|received (?:the )?(?:wrong|incorrect))\b",
    "not_as_described": r"\b(?:not as (?:described|advertised|pictured|shown|stated)|does not match|doesn['’]?t match|different from (?:the )?(?:picture|pictures|photo|photos|description|image|images)|nothing like (?:the )?(?:picture|photo|description|image)|misleading|false advertising|fake|counterfeit|knock[- ]?off)\b",
    "size_fit": r"\b(?:too (?:small|big|large|tight|loose|short|snug)|(?:is|are|was|were|way|bit|little|much|slightly|just|far) too long|runs (?:small|large|big)|(?:does not|doesn['’]?t|didn['’]?t|did not) fit|wrong size|not true to size|sizing (?:is )?(?:off|wrong)|fits? (?:poorly|badly))\b",
    "delivery_shipping": r"\b(?:arrived late|came late|delivered late|delayed|never arrived|never came|never received|did not arrive|didn['’]?t arrive|did not receive|didn['’]?t receive|still waiting|took (?:weeks|forever|too long)|lost in (?:the )?(?:mail|transit|shipping)|arrived (?:damaged|crushed|open|broken|smashed)|(?:box|package|packaging) (?:was )?(?:crushed|damaged|open|torn|smashed|ripped)|poorly packed|poor packaging)\b",
    "customer_service": r"\b(?:customer (?:service|support|care)|support team|(?:seller|company|vendor) (?:never|did not|didn['’]?t|refused|ignored|would not|wouldn['’]?t|was rude|was unhelpful)|no (?:response|reply|answer)|(?:nobody|no one) (?:answers|replies|responds|responded|replied|answered|is answering)|never (?:responded|replied|answered|heard back)|unresponsive|refused (?:to )?(?:refund|replace|return)|still no refund)\b",
}

# ---------------- v2 keyword rules: v1 plus extra phrasings ----------------
_V2_EXTRA = {
    "damaged_defective": r"glitch(?:es|y)?|interrupted|dead|failed|dies|no sound|won['’]?t (?:turn on|charge|work)|(?:doesn['’]?t|does not|didn['’]?t|did not|won['’]?t|will not|can['’]?t|cannot) (?:\w+ ){0,2}(?:turn|power|switch|start|charge|connect|pair|boot)\b|(?:never|didn['’]?t|did not|doesn['’]?t|does not|wouldn['’]?t|would not|not) (?:\w+ ){0,2}work(?:ed|s)?\b|poor(?:ly)? (?:quality|made|built)|(?:cheap(?:ly)?|flimsy) (?:made|material|plastic|quality|feeling)|not (?:very )?durable|burn(?:ed|t) out|snapped|corro\w+|rust(?:ed)?|flicker\w*|hyper ?flash\w*|peel(?:ed|ing)|fell off|falls off|came loose|came off|shut down|warped|barely works|did(?:n['’]?t| not) do what|keeps? (?:popping|falling|slipping|coming) (?:off|out)|coming out of|break(?:s)? (?:away|off)|stretched out|missing (?:\\w+ ){0,2}(?:hardware|posts?|bolts?|screws?|gaskets?)",
    "wrong_item": r"(?:shipped|sent|send) (?:me )?(?:the )?wrong|received (?:a |an )?(?:different|wrong)",
    "not_as_described": r"contrary to|isn['’]?t as (?:pictured|described|advertised)|not as listed|different from what|not (?:what|as) (?:i |we |you |was |it was |they )?(?:expected|promised)|(?:ugly|wrong|different|dull|faded) (?:\w+ )?colou?r|looks? (?:nothing like|completely different|very different)|(?:description|listing|ad|seller) (?:said|claimed|indicated|stated)|(?:said|claimed|stated|advertised|told) (?:it )?(?:would|will|to) (?:fit|work)|not (?:what )?(?:was )?(?:advertised|described)|(?:picture|photo|image) shows|received (?:a |an )?used|unit (?:i )?received was used|(?:much )?lighter than",
    "size_fit": r"(?:smaller|bigger|larger|tighter|looser) than (?:i|we|you|expected|described|pictured|advertised|the (?:photo|picture|listing))|(?:can['’]?t|cannot|could not|couldn['’]?t) (?:wear|fit|put (?:it )?on)|not (?:a )?(?:direct |proper |correct )?fit|not compatible|incompatible|(?:will not|won['’]?t|would not|wouldn['’]?t|does not|doesn['’]?t|did not|didn['’]?t) (?:work|fit) (?:with|on|in)|wrong (?:part|model)|no fit guide|fit(?:ment)? (?:was |is )?(?:not|isn['’]?t|wasn['’]?t) (?:good|accurate|correct|right)|(?:won['’]?t|would not|wouldn['’]?t|cannot|can['’]?t) fit|not fit",
    "delivery_shipping": r"never (?:got|showed up|was delivered|delivered)|not delivered|didn['’]?t (?:get|show up)|did not (?:get|show up)|unpackaged|came unwrapped|(?:box|package) (?:was )?(?:half )?crushed|crushed (?:box|package)|took (?:\w+ ){0,3}(?:weeks?|months?|forever|ages|too long)|(?:late|slow) (?:delivery|shipping)|shipping (?:took|was (?:slow|late))",
    "customer_service": r"(?:no way to|unable to|cannot|can['’]?t|couldn['’]?t) (?:contact|reach)|no refund|never (?:acknowledged|answered|replied|responded)|(?:ignored|ignoring) (?:me|my|us)|won['’]?t (?:do|accept) (?:a )?return|not returnable|non-returnable|replacement.{0,40}(?:nothing|never|not) (?:has )?(?:come|arrived)",
}
PATTERNS = {}
for _k, _p in PATTERNS_V1.items():
    _extra = _V2_EXTRA[_k]
    PATTERNS[_k] = _p if not _extra else _p[:-3] + "|" + _extra + r")\b"   # reopen the group before the closing )\b

# ---------------- zero-shot hypotheses ----------------
# Zero-shot asks: how well does "This review says that <hypothesis>" follow from the review?
HYP_V1 = {
    "damaged_defective": "the product is damaged, broken or does not work",
    "wrong_item": "the wrong item was delivered",
    "not_as_described": "the product does not match its description or pictures",
    "size_fit": "the size or fit is wrong",
    "delivery_shipping": "delivery was late or lost, or the package was damaged in shipping",
    "customer_service": "the seller or customer service was unhelpful",
}
HYP = {
    "damaged_defective": "the product is damaged, broken, defective or stopped working",
    "wrong_item": "the seller sent a different item than the one ordered",
    "not_as_described": "the product is different from what was advertised or pictured",
    "size_fit": "the part does not fit the vehicle, is the wrong size or is not compatible",
    "delivery_shipping": "the item never arrived, arrived late, or the package was lost or damaged in shipping",
    "customer_service": "the seller or customer support did not help or did not respond",
}

# What the business should do for each issue (used by the priority/triage stage)
ACTIONS = {
    "damaged_defective": "Offer replacement or refund; log for supplier quality review",
    "wrong_item": "Ship the correct item; check warehouse picking",
    "not_as_described": "Review the listing (photos/description); consider refund",
    "size_fit": "Offer an exchange for the right part; check the vehicle fitment data",
    "delivery_shipping": "Check carrier/packaging; offer compensation if lost or late",
    "customer_service": "Escalate to a support lead for service recovery",
}


_MOJIBAKE = (("â€™", "'"), ("â€˜", "'"), ("â€œ", '"'), ("â€\x9d", '"'), ("â€“", "-"), ("â€”", "-"), ("â€¦", "..."), ("Â ", " "))


def fix_text(t):
    """Repairs text that was saved with the wrong encoding (for example "itâ€™s" -> "it's"); common in scraped reviews."""
    t = str(t)
    for a, b in _MOJIBAKE:
        t = t.replace(a, b)
    return t


def clean(t):
    t = fix_text(t).lower()
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\[\[[a-z]+:[^\]]*\]\]", " ", t)                 # Amazon media placeholders such as [[VIDEOID:...]]
    return re.sub(r"\s+", " ", t).strip()


def rule_flags(texts, patterns=None):
    """Boolean DataFrame (one column per issue): which keyword rules fire. Default = v2 rules."""
    patterns = patterns or PATTERNS
    s = pd.Series([clean(t) for t in texts])
    return pd.DataFrame({k: s.str.contains(patterns[k], regex=True, case=False) for k in ISSUES})


_zs = None


def zs_scores(texts, hyp=None, batch=32):
    """Zero-shot probability (0..1) per issue for each review. Needs `transformers`; much faster on a GPU."""
    global _zs
    import torch
    from transformers import pipeline
    hyp = hyp or HYP
    if _zs is None:
        dev = 0 if torch.cuda.is_available() else -1
        try:
            _zs = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", device=dev)
        except Exception:
            _zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=dev)
    texts = [str(t)[:900] for t in texts]
    out = {k: [] for k in ISSUES}
    for i in range(0, len(texts), batch):
        res = _zs(texts[i:i + batch], candidate_labels=list(hyp.values()),
                  hypothesis_template="This review says that {}.", multi_label=True, batch_size=16)
        if isinstance(res, dict):
            res = [res]
        for r in res:
            d = dict(zip(r["labels"], r["scores"]))
            for k in ISSUES:
                out[k].append(d[hyp[k]])
    return pd.DataFrame(out)


def detect_issues(texts, config):
    """config = {issue: {"method": "rules" | "zs" | "either", "thr": float}} -> boolean DataFrame of flagged issues."""
    rules = rule_flags(texts)
    zs = zs_scores(texts) if any(v["method"] != "rules" for v in config.values()) else None
    out = {}
    for k, v in config.items():
        r = rules[k].to_numpy(bool)
        z = (zs[k].to_numpy() > v.get("thr", 1.0)) if zs is not None else np.zeros(len(r), bool)
        out[k] = r if v["method"] == "rules" else (z if v["method"] == "zs" else (r | z))
    return pd.DataFrame(out)


# ---------------- which PART is the review about? ----------------
# A transparent keyword lexicon of auto parts and accessories, grouped like the retailer's catalogue. More specific entries come first, and a
# matched phrase is "claimed" so that a more general entry cannot match inside it ("tire pressure gauge" is a gauge, not a tyre).
# It is NOT a trained model: it names the part when the review uses an ordinary word for it, and says nothing otherwise.
PARTS = [
    # (key, name shown, catalogue group, regex)
    ("tire_gauge", "Tyre pressure gauge", "Tools & equipment", r"(?:tire|tyre)\s+(?:pressure\s+)?gauges?|pressure gauges?"),
    ("inflator", "Tyre inflator / air compressor", "Tools & equipment", r"(?:tire|tyre)\s+inflators?|inflators?|air compressors?|portable compressors?|compressors?"),
    ("tire_repair", "Tyre repair kit", "Tires & wheels", r"(?:tire|tyre)\s+(?:repair|plug|patch)\w*|plug kits?|patch kits?"),
    ("tpms", "Tyre pressure sensor (TPMS)", "Tires & wheels", r"tpms|valve stems?|valve caps?"),
    ("lug_nuts", "Lug nuts / wheel locks", "Tires & wheels", r"lug nuts?|lug bolts?|wheel locks?|wheel studs?|lug wrench(?:es)?"),
    ("fasteners", "Screws, bolts & nuts", "Replacement parts", r"screws?|bolts?|nuts?|washers?|rivets?|fasteners?|self-?tapping"),
    ("steering_cover", "Steering wheel cover", "Interior accessories", r"steering wheel covers?|steering covers?"),
    ("steering_wheel", "Steering wheels", "Interior accessories", r"steering wheels?"),
    ("tires", "Tyres", "Tires & wheels", r"tires?|tyres?"),
    ("wheels", "Wheels & rims", "Tires & wheels", r"wheels?|rims?|hub ?caps?|wheel covers?|center caps?|wheel spacers?|spacers?"),
    ("tail_lights", "Tail / brake lights", "Lights & lighting", r"tail ?lights?|tail ?lamps?|brake lights?|reverse lights?|backup lights?"),
    ("headlights", "Headlights", "Lights & lighting", r"headlights?|head lights?|headlamps?|xenon|hid kits?|projectors?|halogen"),
    ("fog_lights", "Fog / driving lights", "Lights & lighting", r"fog ?lights?|driving lights?|fog lamps?"),
    ("light_bar", "Light bars / work lights", "Lights & lighting", r"light bars?|work lights?|off-?road lights?|led pods?"),
    ("turn_signal", "Turn signals / flashers", "Lights & lighting", r"turn signals?|flashers?|blinkers?|hyper ?flash\w*|signal lights?"),
    ("interior_lights", "Interior lights", "Lights & lighting", r"dome lights?|interior lights?|map lights?|ambient lights?"),
    ("warning_light", "Dashboard warning light", "Dashboard & electronics", r"check engine lights?|engine lights?|warning lights?|abs lights?|dash lights?|airbag lights?"),
    ("bulbs", "Bulbs / LEDs", "Lights & lighting", r"led bulbs?|light bulbs?|bulbs?|led lights?|leds?|lights?|lamps?"),
    ("brakes", "Brakes", "Replacement parts", r"brake pads?|brake rotors?|brake calipers?|brake lines?|brake shoes?|brake boosters?|rotors?|calipers?|brakes?"),
    ("filters", "Filters", "Replacement parts", r"(?:oil|air|cabin|fuel|transmission) filters?|filters?"),
    ("spark_plugs", "Spark plugs / ignition coils", "Replacement parts", r"spark plugs?|ignition coils?|coil packs?|coils?"),
    ("jump_starter", "Jump starter / battery charger", "Tools & equipment", r"jump starters?|jump boxes?|battery chargers?|battery tenders?|battery maintainers?|booster packs?|jumper cables?|jump cables?|chargers?|inverters?"),
    ("battery", "Battery", "Replacement parts", r"car batter(?:y|ies)|batter(?:y|ies)"),
    ("alternator", "Alternator / starter", "Replacement parts", r"alternators?|starter motors?|starters?"),
    ("seat_belt", "Seat belts", "Interior accessories", r"seat ?belts?"),
    ("belts_hoses", "Belts & hoses", "Replacement parts", r"serpentine belts?|timing belts?|drive belts?|belts?|radiator hoses?|hoses?"),
    ("gaskets", "Gaskets & seals", "Replacement parts", r"gaskets?|o-?rings?|valve covers?|seals?"),
    ("sensors", "Sensors", "Replacement parts", r"o2 sensors?|oxygen sensors?|abs sensors?|map sensors?|maf sensors?|sensors?"),
    ("suspension", "Suspension (control arms, shocks, struts)", "Replacement parts", r"control arms?|ball joints?|tie rods?|sway bars?|bushings?|struts?|shocks?|leaf springs?|lift kits?|leveling kits?|springs?"),
    ("cooling", "Cooling (radiator, water pump)", "Replacement parts", r"radiators?|thermostats?|water pumps?|cooling fans?|fan clutch|heater cores?"),
    ("exhaust", "Exhaust", "Replacement parts", r"exhaust(?: tips?)?|mufflers?|catalytic converters?|tailpipes?|downpipes?|headers?"),
    ("fuel_system", "Fuel system", "Replacement parts", r"fuel pumps?|fuel injectors?|fuel lines?|gas caps?|fuel caps?|carburetors?|fuel tanks?"),
    ("drivetrain", "Clutch / axle / drivetrain", "Replacement parts", r"clutch(?:es)?|transmission|axles?|cv joints?|driveshaft|differentials?"),
    ("locks_fobs", "Locks, handles & key fobs", "Replacement parts", r"door locks?|door handles?|window regulators?|latches?|hinges?|key fobs?|fobs?|remotes?|actuators?"),
    ("electrical", "Switches, wiring & connectors", "Replacement parts", r"wiring harness(?:es)?|harness(?:es)?|switches?|relays?|fuses?|connectors?|wires?|wiring|cables?|plugs?"),
    ("modules", "Electronic modules / ECU", "Replacement parts", r"ecus?|ecms?|control modules?|ignition modules?|modules?|circuit boards?|transistors?"),
    ("wipers", "Wiper blades", "Exterior accessories", r"wiper blades?|windshield wipers?|wipers?|(?:replacement |rain |silicone |beam )?blades?"),   # bare "blades" in an automotive review means wiper blades
    ("car_cover", "Car / truck covers", "Exterior accessories", r"car covers?|vehicle covers?|truck covers?|tonneau(?: covers?)?|bed covers?"),
    ("running_boards", "Running boards / bumpers / flares", "Exterior accessories", r"mud flaps?|splash guards?|running boards?|side steps?|nerf bars?|step bars?|bull bars?|grille guards?|bumpers?|grilles?|fender flares?|flares?"),
    ("racks_hitches", "Racks & hitches", "Exterior accessories", r"roof racks?|cross ?bars?|cargo carriers?|bike racks?|trailer hitch(?:es)?|hitch(?:es)?|receivers?|tow hooks?|tow straps?|tow bars?|ball mounts?|racks?"),
    ("mirrors", "Mirrors", "Exterior accessories", r"side mirrors?|mirror covers?|mirrors?"),
    ("emblems", "Emblems, decals & plate frames", "Exterior accessories", r"emblems?|badges?|decals?|stickers?|vinyl|license plate frames?|plate frames?|plate covers?|license plates?"),
    ("visors", "Window visors / rain guards", "Exterior accessories", r"window visors?|vent visors?|rain guards?|wind deflectors?|vent shades?"),
    ("bed_liner", "Truck bed accessories", "Exterior accessories", r"bed liners?|truck bed|tailgates?"),
    ("antenna", "Antennas", "Exterior accessories", r"antennas?|antennae"),
    ("trim", "Trim & weatherstripping", "Exterior accessories", r"weather ?strip(?:ping)?|moldings?|door edge guards?|trim"),
    ("audio", "Car audio", "Electronics", r"car stereos?|stereos?|speakers?|subwoofers?|head units?|amplifiers?|car radios?|radios?"),
    ("sunshade", "Sunshades", "Interior accessories", r"sun ?shades?|windshield shades?|sun visors?|window shades?|windshield covers?"),
    ("seat_covers", "Seat covers & cushions", "Interior accessories", r"seat covers?|seat cushions?|seat protectors?|seat pads?|seat gap fillers?|gap fillers?"),
    ("floor_mats", "Floor mats & liners", "Interior accessories", r"floor mats?|floor liners?|car mats?|cargo liners?|trunk mats?|all-?weather mats?|mats?"),
    ("phone_mount", "Phone mounts", "Interior accessories", r"phone mounts?|phone holders?|phone cradles?|car mounts?|dash mounts?|tablet mounts?"),
    ("dash_cam", "Dash cams & cameras", "Interior accessories", r"dash ?cams?|backup cameras?|rear cameras?|cameras?"),
    ("dash_cover", "Dash covers", "Interior accessories", r"dash covers?|dash mats?|dashboard covers?|dashboards?"),
    ("organizers", "Organizers & cup holders", "Interior accessories", r"cup holders?|organizers?|center consoles?|consoles?|trunk organizers?"),
    ("shifter_pedals", "Shift knobs & pedals", "Interior accessories", r"shift knobs?|gear shifts?|shifters?|pedal covers?|pedals?"),
    ("obd", "OBD scanners / code readers", "Tools & equipment", r"obd ?i*2?|code readers?|scan tools?|scanners?|diagnostic(?:s| tools?)?"),
    ("jacks", "Jacks & stands", "Tools & equipment", r"jack stands?|jacks?|ramps?|creepers?"),
    ("hand_tools", "Hand tools", "Tools & equipment", r"torque wrenches?|wrench(?:es)?|sockets?|ratchets?|pliers|screwdrivers?|tool kits?|tool sets?|pullers?|flaring tools?|multimeters?|tools?"),
    ("vacuum", "Car vacuum", "Car care", r"vacuums?"),
    ("wax_polish", "Wax, polish & coatings", "Car care", r"waxes|wax|polish(?:es)?|sealants?|ceramic coatings?|ceramic|quick detailers?|detailers?|compounds?"),
    ("cleaners", "Cleaners & wash", "Car care", r"cleaners?|car wash|washes|wash|soap|degreasers?|shampoo|tire shine|tire dressing"),
    ("detailing", "Detailing tools", "Car care", r"microfiber|towels?|applicators?|brushes?|sponges?|clay bars?|buffers?|polishers?"),
    ("air_freshener", "Air fresheners", "Car care", r"air fresheners?|fresheners?|fragrance|scent"),
    ("paint", "Paint & body supplies", "Paint & paint supplies", r"touch-?up paint|spray paint|paint|primers?|clear coats?|sandpaper|body filler|bondo|masking tape|rust converter|undercoating"),
    ("washer_fluid", "Washer fluid", "Oils & fluids", r"washer fluid|windshield washer"),
    ("coolant", "Coolant / antifreeze", "Oils & fluids", r"coolant|antifreeze"),
    ("other_fluids", "Brake / transmission / steering fluid", "Oils & fluids", r"brake fluid|transmission fluid|power steering fluid|atf|gear oil"),
    ("additives", "Fuel additives & lubricants", "Oils & fluids", r"fuel additives?|fuel treatment|injector cleaner|octane booster|stabilizer|additives?|grease|lubricants?|wd-?40"),
    ("motor_oil", "Motor oil", "Oils & fluids", r"motor oil|engine oil|oil change|synthetic oil|oil"),
    ("rv", "RV / trailer parts", "RV parts & accessories", r"awnings?|leveling blocks?|levelers?|slide-?outs?|sewer hoses?|rv (?:parts?|hoses?|batter(?:y|ies)|covers?|water filters?)"),
    ("motorcycle", "Motorcycle / powersports parts", "Motorcycle & powersports", r"helmets?|handlebars?|saddle ?bags?|sprockets?|fairings?|motorcycle (?:covers?|chains?|mirrors?|parts?|batter(?:y|ies))|atv (?:parts?|winch(?:es)?|plows?)"),
    ("performance", "Performance parts", "Performance parts", r"cold air intakes?|intakes?|tuners?|superchargers?|spoilers?|performance"),
]
_PARTS_COMPILED = [(k, n, g, re.compile(r"\b(?:" + p + r")\b", flags=re.I)) for k, n, g, p in PARTS]


def mentioned_parts(text, n=3):
    """Parts a review talks about, in the order they first appear: [{'key', 'name', 'group', 'phrase'}]. Empty list = no part named."""
    t = clean(text)
    claimed, found = [], []
    for k, name, grp, rx in _PARTS_COMPILED:
        for m in rx.finditer(t):
            if any(m.start() < e and m.end() > s for s, e in claimed):
                continue
            claimed.append((m.start(), m.end()))
            found.append((m.start(), {"key": k, "name": name, "group": grp, "phrase": m.group(0)}))
            break
    found.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _, d in found:
        if d["key"] not in seen:
            seen.add(d["key"])
            out.append(d)
    return out[:n]


# Which of the photo model's seven object types could plausibly depict the part a review names? Used ONLY to notice a photo that does not seem to
# belong to the review. A part that is not listed here (tyres, wipers, brakes, bulbs ...) cannot be shown by any of the seven types.
PHOTO_FOR = {
    "fasteners": {"screw", "metal nut"}, "lug_nuts": {"screw", "metal nut"},
    "electrical": {"cable", "transistor"}, "modules": {"transistor"}, "sensors": {"transistor"}, "obd": {"transistor"}, "jump_starter": {"cable"},
    "seat_covers": {"leather"}, "steering_cover": {"leather"}, "floor_mats": {"carpet"},
    "motor_oil": {"bottle"}, "coolant": {"bottle"}, "washer_fluid": {"bottle"}, "other_fluids": {"bottle"}, "additives": {"bottle"},
    "cleaners": {"bottle"}, "wax_polish": {"bottle"}, "air_freshener": {"bottle"}, "paint": {"bottle"},
}
