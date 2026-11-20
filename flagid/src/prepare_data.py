"""Stage 1 data prep: flag SVGs -> labels.json + PNG cache, and COCO -> backgrounds.

Usage:
    python src/prepare_data.py flags        # needs data/_downloads/country-flags (git clone)
    python src/prepare_data.py backgrounds  # needs data/_downloads/val2017 (COCO unzip)
"""
import json
import random
import shutil
import sys

import cairosvg
from PIL import Image
from tqdm import tqdm

from config import CFG

# 193 UN member states + 2 observer states (VA = Holy See, PS = Palestine). ISO 3166-1 alpha-2.
UN_CODES = """
AF AL DZ AD AO AG AR AM AU AT AZ BS BH BD BB BY BE BZ BJ BT BO BA BW BR BN BG BF BI CV KH CM CA
CF TD CL CN CO KM CG CD CR CI HR CU CY CZ DK DJ DM DO EC EG SV GQ ER EE SZ ET FJ FI FR GA GM GE
DE GH GR GD GT GN GW GY HT HN HU IS IN ID IR IQ IE IL IT JM JP JO KZ KE KI KP KR KW KG LA LV LB
LS LR LY LI LT LU MG MW MY MV ML MT MH MR MU MX FM MD MC MN ME MA MZ MM NA NR NP NL NZ NI NE NG
MK NO OM PK PW PA PG PY PE PH PL PT QA RO RU RW KN LC VC WS SM ST SA SN RS SC SL SG SK SI SB SO
ZA SS ES LK SD SR SE CH SY TJ TZ TH TL TG TO TT TN TR TM TV UG UA AE GB US UY UZ VU VE VN YE ZM
ZW VA PS
""".split()


def prepare_flags():
    p = CFG.paths
    src = p.downloads / "country-flags"
    names = json.loads((src / "countries.json").read_text())
    codes = sorted(set(UN_CODES))
    assert len(codes) == len(UN_CODES) == 195, f"expected 195 unique codes, got {len(codes)}"

    p.flags_svg.mkdir(parents=True, exist_ok=True)
    p.flags_png.mkdir(parents=True, exist_ok=True)
    labels = {}
    for idx, code in enumerate(tqdm(codes, desc="flags")):
        iso2 = code.lower()
        svg_src = src / "svg" / f"{iso2}.svg"
        assert svg_src.exists(), f"missing SVG for {code}"
        svg_dst = p.flags_svg / f"{iso2}.svg"
        shutil.copyfile(svg_src, svg_dst)
        png = p.flags_png / f"{iso2}.png"
        cairosvg.svg2png(url=str(svg_dst), write_to=str(png), output_width=CFG.data.raster_width)
        labels[idx] = {"iso2": iso2, "name": names[code]}
    if CFG.data.include_no_flag:
        labels[len(codes)] = {"iso2": CFG.data.no_flag_label, "name": "No flag"}
    p.labels.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    print(f"wrote {len(labels)} classes to {p.labels}")


def prepare_backgrounds():
    p, d = CFG.paths, CFG.data
    src_imgs = sorted((p.downloads / "val2017").glob("*.jpg"))
    rng = random.Random(0)
    rng.shuffle(src_imgs)
    p.backgrounds.mkdir(parents=True, exist_ok=True)
    for f in tqdm(src_imgs[: d.num_backgrounds], desc="backgrounds"):
        im = Image.open(f).convert("RGB")
        im.thumbnail((d.background_max_side, d.background_max_side))
        im.save(p.backgrounds / f.name, quality=92)
    print(f"{len(list(p.backgrounds.glob('*.jpg')))} backgrounds in {p.backgrounds}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "flags"
    {"flags": prepare_flags, "backgrounds": prepare_backgrounds}[step]()
