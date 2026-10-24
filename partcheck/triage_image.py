"""Image evidence for the triage engine: is the customer's photo showing damage, or the wrong item?

Two families of methods (both built on a pretrained CLIP image encoder via sentence-transformers):
  * TEXT-based (zero-shot): compare the photo with the words "a damaged <product>" / "a photo of a <product>".
  * REFERENCE-based: compare the photo with REFERENCE PHOTOS of the ordered product (what a retailer has in its catalogue):
      - ref_match_prob  : is the photo closest to the ordered product's reference photos (vs other products)?  -> wrong item
      - anomaly_scores  : how far is the photo from the product's GOOD reference photos?                       -> damage
Image risk = max(P(damaged), 1 - P(match)) is fed to triage_engine.priority_score.
"""
import numpy as np

LOGIT_SCALE = 100.0   # CLIP's learned temperature


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class ClipScorer:
    def __init__(self, name="clip-ViT-B-32"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def embed_images(self, images, batch=32):
        return self.m.encode(list(images), batch_size=batch, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)

    def embed_texts(self, texts):
        return self.m.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    # ---- text-based (zero-shot) ----
    def defect_prob(self, img_emb, products):
        """Zero-shot probability that the photo shows a damaged/defective item. `products` = name per image."""
        products = list(products)
        out = np.zeros(len(products))
        for p in sorted(set(products)):
            idx = [i for i, q in enumerate(products) if q == p]
            t = self.embed_texts([f"a photo of a damaged, broken or defective {p}", f"a photo of a {p} in good condition"])
            out[idx] = _softmax(LOGIT_SCALE * img_emb[idx] @ t.T)[:, 0]
        return out

    def match_prob(self, img_emb, ordered, universe):
        """Probability (among all candidate product names, by TEXT) that the photo shows the ORDERED product."""
        universe = list(universe)
        t = self.embed_texts([f"a photo of a {u}" for u in universe])
        P = _softmax(LOGIT_SCALE * img_emb @ t.T)
        pos = {u: i for i, u in enumerate(universe)}
        return np.array([P[i, pos[o]] for i, o in enumerate(ordered)])


# ---------------- reference-based (catalogue photos) ----------------
def make_prototypes(refs):
    """refs: {product: array of reference-photo embeddings} -> {product: normalised mean embedding}."""
    out = {}
    for o, R in refs.items():
        v = R.mean(axis=0)
        out[o] = v / np.linalg.norm(v)
    return out


def ref_match_prob(img_emb, ordered, protos, temp=50.0):
    """Probability that the photo shows the ORDERED product, judged against every product's reference photos."""
    names = list(protos)
    P = np.stack([protos[n] for n in names])
    S = _softmax(temp * img_emb @ P.T)
    pos = {n: i for i, n in enumerate(names)}
    return np.array([S[i, pos[o]] for i, o in enumerate(ordered)])


def anomaly_scores(img_emb, products, refs, exclude_self=False):
    """1 - (highest cosine similarity to a GOOD reference photo of the same product). Higher = more unlike a good item."""
    products = list(products)
    out = np.zeros(len(products))
    for o in sorted(set(products)):
        idx = [i for i, p in enumerate(products) if p == o]
        S = img_emb[idx] @ refs[o].T
        if exclude_self:                       # a reference photo must not be compared with itself
            S = np.where(S > 0.9999, -np.inf, S)
        out[idx] = 1 - S.max(axis=1)
    return out


def image_risk(p_defect, p_match):
    """0..1 risk from the photo: damaged OR not the ordered product."""
    return np.maximum(np.asarray(p_defect), 1 - np.asarray(p_match))


class ImageEvidence:
    """Bundles everything needed to score a customer photo: which product it shows, damage probability, product-match probability, image risk."""

    def __init__(self, clip, refs, protos, platt, lr, mode, products, damage_thr=0.5, anom_thr=None, known_margin=0.05):
        self.clip, self.refs, self.protos, self.platt, self.lr, self.mode, self.products = clip, refs, protos, platt, lr, mode, list(products)
        self.damage_thr = float(damage_thr)      # (older artifacts) damage probability above which a photo is treated as showing damage
        # Per-product damage cut-off on the RAW anomaly score (1 - similarity to the nearest good reference photo), chosen on the training half.
        # None for a product = no cut-off reached the precision target, so the damage check is OFF for that part type.
        self.anom_thr = dict(anom_thr) if anom_thr is not None else None
        self.known_margin = float(known_margin)  # a photo must be within this margin of looking like a normal photo of SOME known product
        # how similar a GOOD photo of each product is to that product's prototype (2nd percentile): a photo far below this is not that product
        self.cmin = {o: float(np.percentile(refs[o] @ protos[o], 2)) for o in refs if o in protos}

    def damage_prob(self, emb, prods):
        prods = list(prods)
        if self.mode == "lr":
            return self.lr.predict_proba(emb)[:, 1]
        if self.mode == "ref" and all(p in self.refs for p in prods):
            return self.platt.predict_proba(anomaly_scores(emb, prods, self.refs).reshape(-1, 1))[:, 1]
        return self.clip.defect_prob(emb, prods)

    def damage_checked(self, product):
        """Is the damage check switched on for this part type? (older artifacts without per-product cut-offs: always on)"""
        return True if self.anom_thr is None else self.anom_thr.get(product) is not None

    def damage_signal(self, emb, prods):
        """Damage risk 0..1. 0 below the part type's cut-off (or when the check is off), 0.5 at the cut-off, rising to 1 at twice the cut-off."""
        prods = list(prods)
        if self.anom_thr is None or not all(p in self.refs for p in prods):
            return self.damage_risk(self.damage_prob(emb, prods))
        a = anomaly_scores(emb, prods, self.refs)
        thr = np.array([self.anom_thr.get(p) if self.anom_thr.get(p) is not None else np.inf for p in prods], dtype=float)
        risk = 0.5 + 0.5 * np.clip((a - thr) / np.maximum(np.where(np.isfinite(thr), thr, 1.0), 0.01), 0.0, 1.0)
        return np.where(a >= thr, risk, 0.0)

    def damage_risk(self, pdm):
        """Rescale a damage probability so that anything below the chosen threshold contributes no risk."""
        return np.clip((np.asarray(pdm) - self.damage_thr) / max(1 - self.damage_thr, 1e-6), 0.0, 1.0)

    def match_prob(self, emb, prods):
        """Probability the photo shows the ORDERED product. Two conditions: (1) it must look like one of the known products at all
        (so unrelated photos are rejected), and (2) it must be closer to the ordered product than to the other products.
        A DAMAGED part still looks like the right part, so it is not mistaken for a wrong item (an earlier version compared the photo with a
        'normal photo' similarity floor, which pushed badly damaged parts to zero and labelled them 'wrong item')."""
        prods = list(prods)
        if not all(p in self.protos for p in prods):
            return np.ones(len(prods))          # unknown product: no mismatch evidence
        rel = ref_match_prob(emb, prods, self.protos)
        return rel * self.known(emb, prods).astype(float)

    def match_prob_v1(self, emb, prods):
        """The earlier rule (relative x absolute similarity floor); kept only to measure the improvement."""
        prods = list(prods)
        rel = ref_match_prob(emb, prods, self.protos)
        cos = np.array([float(emb[i] @ self.protos[o]) for i, o in enumerate(prods)])
        floor = np.array([self.cmin.get(o, 0.0) for o in prods])
        return rel * np.clip((cos - (floor - 0.10)) / 0.10, 0.0, 1.0)

    def known(self, emb, prods=None):
        """Does each photo look like a known part? Yes if it is close to a normal photo of some known part (within `known_margin` of the floor),
        or, when the ordered part is named, if it is clearly closest to THAT part (relative match >= 0.99): a badly damaged part still looks like itself.
        Measured on all held-out photos: 0.7% of defective photos fail this test, and 0 of 876 photos of objects outside the catalogue pass it."""
        names = list(self.protos)
        P = np.stack([self.protos[n] for n in names])
        S = emb @ P.T
        j = S.argmax(axis=1)
        floor = np.array([self.cmin.get(names[k], 0.0) for k in j])
        ok = S[np.arange(len(emb)), j] >= floor - self.known_margin
        if prods is not None:
            prods = list(prods)
            idx = np.array([names.index(p) if p in names else -1 for p in prods])
            rel = ref_match_prob(emb, prods, self.protos)
            ok = ok | ((j == idx) & (rel >= 0.99))
        return ok

    def identify(self, emb):
        """Which of the known products does each photo show? Returns [(product or None, similarity)]; None = looks like none of them."""
        names = list(self.protos)
        P = np.stack([self.protos[n] for n in names])
        out = []
        for row in emb @ P.T:
            j = int(row.argmax())
            n = names[j]
            out.append((n, float(row[j])) if row[j] >= self.cmin.get(n, 0.0) - self.known_margin else (None, float(row[j])))
        return out

    def risk_from_emb(self, emb, prods):
        """Image risk when the ordered product is known: damaged OR not the ordered product."""
        pdm, pm = self.damage_prob(emb, prods), self.match_prob(emb, prods)
        return np.maximum(self.damage_signal(emb, prods), 1 - pm), pdm, pm

    def risk(self, images, prods):
        return self.risk_from_emb(self.clip.embed_images(images), prods)

    def save(self, path):
        import joblib
        joblib.dump({"refs": self.refs, "protos": self.protos, "platt": self.platt, "lr": self.lr,
                     "mode": self.mode, "products": self.products, "damage_thr": self.damage_thr,
                     "anom_thr": self.anom_thr, "known_margin": self.known_margin}, path)

    @classmethod
    def load(cls, path, clip=None):
        import joblib
        d = joblib.load(path)
        return cls(clip or ClipScorer(), d["refs"], d["protos"], d["platt"], d["lr"], d["mode"], d["products"], d.get("damage_thr", 0.5),
                   d.get("anom_thr"), d.get("known_margin", 0.05))


# ---------------- "what is in this picture?": zero-shot recognition of auto-part pictures ----------------
# A customer's photo can show a tyre, a bottle of oil, a wiper blade ... or something that has nothing to do with the order (food, a pet, a person,
# a screenshot). This step names what the picture shows, so the tool can (1) reject a random picture and (2) check the picture against the part the
# review is about. It needs NO training photos: CLIP compares the picture with short text descriptions. Its accuracy is measured in the notebook.
#
# (key, name shown, descriptions CLIP compares the picture with, review-lexicon keys (triage_core.PARTS) that this picture class can depict)
PHOTO_CLASSES = [
    ("tire", "tyres", ["a car tire", "a tyre"], ["tires", "tire_repair", "tpms"]),
    ("wheel", "wheels & rims", ["a car wheel rim", "an alloy wheel"], ["wheels", "lug_nuts"]),
    ("wiper", "wiper blades", ["a windshield wiper blade", "car wiper blades"], ["wipers"]),
    ("brake", "brake parts", ["a brake pad", "a brake rotor disc"], ["brakes"]),
    ("light", "lights & bulbs", ["a car headlight", "a car tail light", "an led light bulb for a car", "a light bar"],
     ["headlights", "tail_lights", "bulbs", "fog_lights", "light_bar", "turn_signal", "interior_lights", "warning_light"]),
    ("battery", "batteries & chargers", ["a car battery", "a battery charger or jump starter"], ["battery", "jump_starter", "alternator"]),
    ("floor_mat", "floor mats", ["car floor mats", "a rubber car floor liner"], ["floor_mats"]),
    ("seat_cover", "seat covers", ["a car seat cover", "a steering wheel cover"], ["seat_covers", "steering_cover", "seat_belt"]),
    ("sunshade", "sunshades", ["a windshield sun shade", "a car window shade"], ["sunshade", "visors"]),
    ("bottle", "bottles, cans & fluids", ["a bottle of motor oil", "a spray can or bottle of car cleaner", "a bottle of car wax or polish", "a can of paint"],
     ["motor_oil", "coolant", "other_fluids", "additives", "washer_fluid", "wax_polish", "cleaners", "air_freshener", "paint", "detailing"]),
    ("filter", "filters", ["an oil filter", "a car air filter"], ["filters"]),
    ("plug", "spark plugs & ignition", ["a spark plug", "an ignition coil"], ["spark_plugs"]),
    ("hose_belt", "belts & hoses", ["a car belt", "a rubber hose"], ["belts_hoses"]),
    ("tools", "tools", ["a wrench or socket set", "a car jack", "a hand tool"], ["hand_tools", "jacks", "vacuum", "tire_gauge", "inflator"]),
    ("gadget", "scanners & gadgets", ["an obd2 car scanner", "a dash cam", "a phone holder mounted in a car", "a car electronic device"],
     ["obd", "dash_cam", "phone_mount", "modules", "sensors", "antenna", "tire_gauge", "inflator"]),
    ("cable", "cables & wiring", ["electrical wires", "a wiring harness", "a cable"], ["electrical"]),
    ("fastener", "screws, bolts & nuts", ["screws and bolts", "nuts and washers", "a bag of fasteners"], ["fasteners", "lug_nuts"]),
    ("mirror", "mirrors", ["a car side mirror", "a rear view mirror"], ["mirrors"]),
    ("emblem", "emblems & plate frames", ["a car emblem or decal", "a license plate frame", "a car sticker"], ["emblems"]),
    ("rack", "racks & hitches", ["a roof rack", "a trailer hitch", "a tow hook or tow strap"], ["racks_hitches"]),
    ("exterior", "bumpers & exterior trim", ["a car bumper", "a running board", "fender flares", "a car cover"],
     ["running_boards", "trim", "bed_liner", "car_cover"]),
    ("suspension", "suspension parts", ["a control arm", "a shock absorber", "a ball joint"], ["suspension"]),
    ("engine", "engine parts", ["a car radiator", "a water pump", "an engine gasket", "an exhaust pipe", "a fuel pump"],
     ["cooling", "gaskets", "fuel_system", "exhaust", "drivetrain"]),
    ("lock", "locks & key fobs", ["a car key fob", "a door handle or lock"], ["locks_fobs"]),
    ("interior", "interior accessories", ["a car cup holder", "a car seat back organizer", "a gear shift knob", "a dashboard cover"],
     ["organizers", "shifter_pedals", "dash_cover"]),
]
# pictures that are not an auto part at all, or that show a whole vehicle (context, not a part)
PHOTO_NON_PART = [
    ("person", "a person or a selfie"), ("animal", "a pet or an animal"), ("food", "food or a drink on a table"),
    ("outdoors", "an outdoor landscape or a building"), ("document", "a document, a receipt or a phone screenshot"),
    ("room", "a room, furniture or a household item"), ("clothes", "clothing or shoes"), ("toy", "a toy or a gadget that is not a car part"),
    ("music", "a musical instrument such as a piano, a guitar or sheet music"), ("paper", "a book, a magazine, printed text or a page of sheet music"),
    ("electronics", "a laptop, a television or a home electronics device"), ("jewellery", "jewellery, a watch or a handbag"),
    ("kitchen", "kitchen items, a plate or a cup"), ("plant", "a plant, a flower or a garden"), ("sports", "a bicycle, a ball or sports equipment"),
    ("art", "a painting, a poster or a drawing"), ("package", "a cardboard box or a shipping package"), ("cosmetics", "cosmetics, skincare or medicine bottles"),
]
PHOTO_OTHER_EXTRA = {"package": "a package or box", "music": "a musical instrument or sheet music", "paper": "a book or printed page", "electronics": "home electronics", "jewellery": "jewellery or a bag",
                     "kitchen": "kitchen items", "plant": "a plant", "sports": "sports equipment", "art": "a painting or poster", "cosmetics": "cosmetics or medicine"}
T_NAME = 0.30       # the picture is only given a part name when that part holds at least this share of the "which part" probability; otherwise "could not identify a part"
PHOTO_VEHICLE = [("vehicle", "a whole car or truck"), ("vehicle", "the inside of a car or a dashboard"), ("vehicle", "a motorcycle or an atv")]
PHOTO_TEMPLATES = ["a photo of {}", "a close-up photo of {}", "a customer photo of {}"]
PHOTO_FOR_KEYS = {k: {c[0] for c in PHOTO_CLASSES if k in c[3]} for k in {k for c in PHOTO_CLASSES for k in c[3]}}   # lexicon key -> picture classes that can show it
PHOTO_NAME = {c[0]: c[1] for c in PHOTO_CLASSES}
PHOTO_OTHER_NAME = {"person": "a person", "animal": "an animal", "food": "food or a drink", "outdoors": "a landscape or a building", "document": "a document or screenshot",
                    "room": "a room or furniture", "clothes": "clothing", "toy": "a toy or gadget"}
PHOTO_OTHER_NAME.update(PHOTO_OTHER_EXTRA)
# Fine part classes are easy to confuse on real customer photos (a tyre photo vs a wheel photo), so the match check works on broad FAMILIES.
PHOTO_FAMILY = {"tire": "wheels", "wheel": "wheels", "light": "lights", "wiper": "exterior", "mirror": "exterior", "emblem": "exterior", "rack": "exterior",
                "exterior": "exterior", "sunshade": "interior", "floor_mat": "interior", "seat_cover": "interior", "interior": "interior", "bottle": "fluids",
                "battery": "electrical", "cable": "electrical", "gadget": "electrical", "plug": "electrical", "tools": "tools", "fastener": "tools",
                "brake": "mechanical", "filter": "mechanical", "hose_belt": "mechanical", "suspension": "mechanical", "engine": "mechanical", "lock": "mechanical"}
FAMILY_NAME = {"wheels": "tyres & wheels", "lights": "lights", "exterior": "exterior accessories", "interior": "interior accessories", "fluids": "bottles & fluids",
               "electrical": "electrical parts", "tools": "tools & fasteners", "mechanical": "mechanical parts"}
FAMILIES_FOR_KEYS = {k: {PHOTO_FAMILY[c] for c in cs} for k, cs in PHOTO_FOR_KEYS.items()}   # lexicon key -> picture families that can show it


class PhotoRecogniser:
    """Zero-shot: which auto part (or non-part) does the picture show?  predict(img_emb) -> list of dicts."""

    def __init__(self, clip):
        self.clip = clip
        self.labels, self.kinds, texts = [], [], []
        groups = [(c[0], "part", c[2]) for c in PHOTO_CLASSES] + [(k, "other", [d]) for k, d in PHOTO_NON_PART] + [("vehicle", "vehicle", [d for _, d in PHOTO_VEHICLE])]
        spans = []
        for key, kind, descs in groups:
            s = len(texts)
            texts += [t.format(d) for d in descs for t in PHOTO_TEMPLATES]
            spans.append((key, kind, s, len(texts)))
        E = clip.embed_texts(texts)
        T = []
        for key, kind, s, e in spans:
            v = E[s:e].mean(axis=0)
            T.append(v / np.linalg.norm(v))
            self.labels.append(key)
            self.kinds.append(kind)
        self.T = np.stack(T)
        self.kinds = np.array(self.kinds)

    def predict(self, emb):
        P = _softmax(LOGIT_SCALE * np.atleast_2d(emb) @ self.T.T)
        out = []
        for row in P:
            p_part, p_other, p_veh = (float(row[self.kinds == k].sum()) for k in ("part", "other", "vehicle"))
            parts = np.where(self.kinds == "part")[0]
            j = parts[int(row[parts].argmax())]
            top = [self.labels[i] for i in parts[np.argsort(-row[parts])][:3]]
            fam = {}
            for i in parts:
                fam[PHOTO_FAMILY[self.labels[i]]] = fam.get(PHOTO_FAMILY[self.labels[i]], 0.0) + float(row[i] / max(p_part, 1e-9))
            oth = np.where(self.kinds == "other")[0]
            o = self.labels[oth[int(row[oth].argmax())]]
            out.append({"top": self.labels[j], "top_p": float(row[j]), "top3": top, "p_part": p_part, "p_other": p_other, "p_vehicle": p_veh,
                        "share": float(row[j] / max(p_part, 1e-9)), "families": fam, "other_top": o,
                        "named": bool(row[j] / max(p_part, 1e-9) >= T_NAME)})
        return out


# ---------------- visible damage on a customer photo (zero-shot, an extra signal; measured on Colab by colab_picture_eval.py) ----------------
# Compares the picture with "damaged" and "fine" descriptions of the part, the bottle and the package. It is NOT the validated 7-part damage check:
# it is switched on in the app only with a threshold chosen on the Colab evaluation (T_DAMAGE in triage_ui.py).
DAMAGE_PAIRS = [
    (["a cracked, broken or badly damaged car part", "a car part with a visible crack or dent"], ["a new car part in perfect condition", "an undamaged car part"]),
    (["a leaking bottle with spilled liquid", "a crushed or dented bottle or can"], ["an intact sealed bottle", "a new bottle in perfect condition"]),
    (["a crushed, torn or wet damaged cardboard box", "a damaged shipping package"], ["an intact cardboard shipping box", "a neat undamaged package"]),
]


class DamageCues:
    """damage_score(img_emb) -> array in 0..1: the highest 'damaged' probability among the part, bottle and package comparisons."""

    def __init__(self, clip):
        self.pairs = []
        for bad, good in DAMAGE_PAIRS:
            tb = clip.embed_texts([t.format(d) for d in bad for t in PHOTO_TEMPLATES])
            tg = clip.embed_texts([t.format(d) for d in good for t in PHOTO_TEMPLATES])
            vb, vg = tb.mean(axis=0), tg.mean(axis=0)
            self.pairs.append(np.stack([vb / np.linalg.norm(vb), vg / np.linalg.norm(vg)]))

    def damage_score(self, emb):
        emb = np.atleast_2d(emb)
        return np.max([_softmax(LOGIT_SCALE * emb @ T.T)[:, 0] for T in self.pairs], axis=0)
