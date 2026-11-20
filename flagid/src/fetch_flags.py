"""Build the flag catalog: data/flags/catalog.csv + one SVG per flag.

Layout:
    data/flags/catalog.csv                  one row per flag (id, names, category, source, license, ...)
    data/flags/world/<iso2>.svg             UN member/observer states (current 195-class model)
    data/flags/regional/<iso2>/<id>.svg     first-level subdivisions, e.g. regional/es/es-an.svg
    data/flags/preview/<pack>.png           raster preview grid for checking

Regional flags come from Wikidata: the country's "contains administrative territorial entity"
(P150) items that have an ISO 3166-2 code (P300) and a flag image (P41, best rank = current
official flag). Files and their license/author come from Wikimedia Commons.

Usage:
    python src/fetch_flags.py world                 # seed catalog with the 195 country flags
    python src/fetch_flags.py regions es [de us ...] # download a country's regional flags
    python src/fetch_flags.py preview es-regions    # render a check grid of one pack
    python src/fetch_flags.py build-labels world es-regions --out data/labels_world_es.json
        # training label list for these packs (countries keep their labels.json order/indices,
        # new flags are appended) + rasterize missing PNGs into data/flags_png/<id>.png
"""
import argparse
import csv
import html
import io
import json
import re
import shutil
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

import cairosvg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import CFG
from dataset import load_labels

ROOT = CFG.paths.data / "flags"
CATALOG = ROOT / "catalog.csv"
FIELDS = ["id", "name", "category", "country", "iso_3166_2", "wikidata", "pack", "path",
          "source_file", "source_url", "license", "author", "attribution_required", "lookalike_group"]
UA = {"User-Agent": "flagid-dataset-builder/0.1 (flag recognition research script)"}
MAX_SVG_BYTES = 5_000_000


def http_get(url, params=None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == 5:
                raise
            wait = max(int(e.headers.get("Retry-After") or 0), 30 * 2 ** attempt)
            print(f"  (rate limited, waiting {wait}s)")
            time.sleep(wait)


def load_catalog():
    if not CATALOG.exists():
        return {}
    with open(CATALOG, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def save_catalog(rows):
    ROOT.mkdir(parents=True, exist_ok=True)
    group_of = {c: "/".join(g) for g in CFG.data.lookalike_groups for c in g}  # config is the source
    for k, r in rows.items():
        r["lookalike_group"] = group_of.get(k, "")
    with open(CATALOG, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for k in sorted(rows, key=lambda k: (rows[k]["category"] != "country", k)):
            w.writerow({f: rows[k].get(f, "") for f in FIELDS})


def seed_world():
    rows = load_catalog()
    out = ROOT / "world"
    out.mkdir(parents=True, exist_ok=True)
    for l in load_labels():
        c = l["iso2"]
        shutil.copy2(CFG.paths.flags_svg / f"{c}.svg", out / f"{c}.svg")
        rows[c] = {**rows.get(c, {}), "id": c, "name": l["name"], "category": "country", "country": c,
                   "pack": "world", "path": f"world/{c}.svg", "source_file": f"svg/{c}.svg",
                   "source_url": "https://github.com/hampusborgos/country-flags", "license": "PD",
                   "author": "hampusborgos/country-flags (from Wikimedia Commons)", "attribution_required": "false"}
    save_catalog(rows)
    print(f"world: {len(load_labels())} country flags -> {out}")


def sparql(query):
    raw = http_get("https://query.wikidata.org/sparql", {"query": query, "format": "json"})
    return json.loads(raw)["results"]["bindings"]


def commons_info(filenames):
    """filename -> {url, license, author, attribution_required, page} (Commons API, 50 per call)."""
    info = {}
    for i in range(0, len(filenames), 50):
        q = json.loads(http_get("https://commons.wikimedia.org/w/api.php", {
            "action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url|extmetadata|mime|size",
            "titles": "|".join("File:" + f for f in filenames[i:i + 50])}))["query"]
        alias = {n["to"]: n["from"] for n in q.get("normalized", [])}
        for p in q["pages"].values():
            if "imageinfo" not in p:
                continue
            ii = p["imageinfo"][0]
            md = ii.get("extmetadata", {})
            text = lambda k: re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", "", md.get(k, {}).get("value", "")))).strip()  # noqa: E731
            name = alias.get(p["title"], p["title"]).removeprefix("File:")
            info[name] = {"url": ii["url"], "mime": ii.get("mime"), "bytes": ii.get("size", 0),
                          "license": text("LicenseShortName"), "author": text("Artist") or text("Credit"),
                          "attribution_required": text("AttributionRequired") or "unknown",
                          "page": ii.get("descriptionurl", "")}
    return info


def fetch_regions(country):
    country = country.lower()
    rows = load_catalog()
    b = sparql(f"""SELECT ?sub ?subLabel ?iso ?flag WHERE {{
        ?c wdt:P297 "{country.upper()}" . ?c wdt:P150 ?sub .
        ?sub wdt:P300 ?iso . ?sub wdt:P41 ?flag .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }} }}""")
    subs = {}
    for x in b:  # an item can have several best-rank flags; keep the first per ISO code
        iso = x["iso"]["value"]
        fname = urllib.parse.unquote(x["flag"]["value"].rsplit("/", 1)[-1])
        subs.setdefault(iso, {"name": x["subLabel"]["value"], "qid": x["sub"]["value"].rsplit("/", 1)[-1],
                              "file": fname})
    if not subs:
        print(f"{country}: no subdivisions with ISO 3166-2 code + flag on Wikidata")
        return
    info = commons_info(sorted({s["file"] for s in subs.values()}))
    out = ROOT / "regional" / country
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for iso, s in sorted(subs.items()):
        fid = iso.lower()
        meta = info.get(s["file"])
        if meta is None or meta["mime"] != "image/svg+xml" or meta["bytes"] > MAX_SVG_BYTES:
            print(f"  SKIP {fid} {s['name']}: {s['file']} ({'missing' if meta is None else meta['mime']})")
            continue
        dst = out / f"{fid}.svg"
        if not dst.exists():
            data = http_get(meta["url"])
            if b"<svg" not in data[:4096]:
                print(f"  SKIP {fid}: not an SVG document")
                continue
            dst.write_bytes(data)
            time.sleep(5.0)  # upload.wikimedia.org throttles bursts hard
        rows[fid] = {**rows.get(fid, {}), "id": fid, "name": s["name"], "category": "region", "country": country,
                     "iso_3166_2": iso, "wikidata": s["qid"], "pack": f"{country}-regions",
                     "path": f"regional/{country}/{fid}.svg", "source_file": s["file"], "source_url": meta["page"],
                     "license": meta["license"], "author": meta["author"][:200],
                     "attribution_required": meta["attribution_required"]}
        ok += 1
        save_catalog(rows)  # after every file, so an interrupted run keeps its progress
        print(f"  {fid:7s} {s['name']:28s} {meta['license']:18s} {s['file']}")
    save_catalog(rows)
    print(f"{country}: {ok}/{len(subs)} regional flags -> {out}")


def preview(pack):
    rows = [r for r in load_catalog().values() if r["pack"] == pack]
    if not rows:
        sys.exit(f"no flags in pack {pack}")
    cols = min(6, len(rows))
    nrows = -(-len(rows) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 2.4, nrows * 1.9), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, sorted(rows, key=lambda r: r["id"])):
        try:
            png = cairosvg.svg2png(url=str(ROOT / r["path"]), output_width=CFG.data.raster_width)
            ax.imshow(Image.open(io.BytesIO(png)).convert("RGBA"))
        except Exception as e:  # noqa: BLE001
            ax.text(0.5, 0.5, f"FAILED\n{type(e).__name__}", ha="center", va="center", color="red")
        ax.set_title(f"{r['id']} {r['name'][:22]}", fontsize=7)
    out = ROOT / "preview" / f"{pack}.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(f"saved {out}")


def build_labels(packs, out):
    import json as _json
    rows = [r for r in load_catalog().values() if r["pack"] in packs]
    base = [l["iso2"] for l in load_labels()] if "world" in packs else []
    order = base + sorted(r["id"] for r in rows if r["id"] not in set(base))
    byid = {r["id"]: r for r in rows}
    for fid in order:
        png = CFG.paths.flags_png / f"{fid}.png"
        if not png.exists():
            cairosvg.svg2png(url=str(ROOT / byid[fid]["path"]), write_to=str(png),
                             output_width=CFG.data.raster_width)
    labels = {str(i): {"iso2": fid, "name": byid[fid]["name"]} for i, fid in enumerate(order)}
    Path(out).write_text(_json.dumps(labels, indent=2, ensure_ascii=False))
    print(f"{len(order)} classes ({len(order) - len(base)} beyond the world set) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("world")
    r = sub.add_parser("regions")
    r.add_argument("countries", nargs="+")
    p = sub.add_parser("preview")
    p.add_argument("pack")
    b = sub.add_parser("build-labels")
    b.add_argument("packs", nargs="+")
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "world":
        seed_world()
    elif a.cmd == "regions":
        for c in a.countries:
            fetch_regions(c)
    elif a.cmd == "build-labels":
        build_labels(a.packs, a.out)
    else:
        preview(a.pack)


if __name__ == "__main__":
    main()
