#!/usr/bin/env python3
"""
xml_to_jsonl.py
Parse a directory of XML files (your example format) and emit one JSON object per <thesis>.
Saves NDJSON (JSONL) to an output file.

Usage:
  python src/parsers/xml_to_jsonl.py --input_dir data/raw_xml --output_file data/jsonl/all.jsonl

Behavior:
- Extracts attributes like ticker and as-of-date if available on <stock-theses> or <thesis>.
- Extracts child elements <reasoning>, <support>, <action>, <indicators> (if present).
- Normalizes 'nan' -> null, trims quotes.
- Adds metadata (source file, index).
"""
import argparse
import os
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime

# Helper: normalize numeric-like strings and 'nan'
def normalize_value(val):
    if val is None:
        return None
    val = val.strip()
    if val.lower() == "nan":
        return None
    # strip wrapping quotes
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        val = val[1:-1]
    return val

def parse_thesis(elem):
    """
    Given an Element for <thesis>, return dict with keys reasoning, support, action, and any indicators.
    """
    data = {}
    # try to pull standard tags
    for tag in ("reasoning", "support", "action"):
        node = elem.find(tag)
        if node is not None and node.text is not None:
            data[tag] = node.text.strip()
        else:
            data[tag] = None
    # capture any other child tags as indicators / features
    indicators = {}
    for child in elem:
        tag = child.tag.lower()
        if tag in ("reasoning", "support", "action"):
            continue
        # try to read numeric or text
        text = child.text.strip() if child.text else None
        if text is not None:
            indicators[tag] = normalize_value(text)
        # also include attributes
        if child.attrib:
            for k, v in child.attrib.items():
                indicators[f"{tag}_{k}"] = normalize_value(v)
    if indicators:
        data["indicators"] = indicators
    return data

def parse_file(path):
    records = []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        print(f"Failed to parse {path}: {e}")
        return records

    # Heuristics: find any 'stock-theses' parent or search for 'thesis' tags
    # handle both <stock-theses ticker="TSLA"> and plain <thesis>
    stock_parents = root.findall(".//stock-theses")
    if not stock_parents:
        # maybe no explicit wrapper; fallback to any thesis
        thesis_nodes = root.findall(".//thesis")
        for idx, th in enumerate(thesis_nodes):
            rec = {}
            rec.update(parse_thesis(th))
            # attempt to find ticker/as-of-date on parent or self
            ticker = th.get("ticker") or root.get("ticker") or None
            as_of = th.get("as-of-date") or th.get("as_of_date") or None
            rec["ticker"] = normalize_value(ticker)
            rec["as_of_date"] = normalize_value(as_of)
            rec["raw_xml_index"] = idx
            records.append(rec)
        return records

    for sp in stock_parents:
        ticker = normalize_value(sp.get("ticker") or sp.get("symbol") or sp.get("stock") or None)
        # find all thesis children inside
        theses = sp.findall(".//thesis")
        if not theses:
            # maybe direct text on stock-theses?
            # create single record from stock-theses
            rec = {"ticker": ticker, "as_of_date": normalize_value(sp.get("as-of-date") or sp.get("as_of_date") or None)}
            # attempt to fill tags
            for tag in ("reasoning", "support", "action"):
                node = sp.find(tag)
                rec[tag] = node.text.strip() if node is not None and node.text else None
            records.append(rec)
            continue

        for idx, th in enumerate(theses):
            rec = {"ticker": ticker}
            # parse content
            rec.update(parse_thesis(th))
            # as-of-date can be on thesis or on parent
            as_of = th.get("as-of-date") or th.get("as_of_date") or sp.get("as-of-date") or sp.get("as_of_date") or None
            rec["as_of_date"] = normalize_value(as_of)
            rec["raw_xml_index"] = idx
            records.append(rec)

    return records

def iso_date_or_none(s):
    if not s:
        return None
    s = s.strip()
    # try common formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            continue
    # try to parse a year-month fragment
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)
    return s  # fallback raw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with XML files")
    parser.add_argument("--output_file", required=True, help="Output NDJSON (jsonl)")
    args = parser.parse_args()

    input_dir = args.input_dir
    out_file = args.output_file
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".xml") or f.lower().endswith(".txt")]
    total = 0
    with open(out_file, "w", encoding="utf-8") as fo:
        for path in files:
            recs = parse_file(path)
            for i, r in enumerate(recs):
                # ensure minimal fields
                r["ticker"] = r.get("ticker") or None
                r["as_of_date"] = iso_date_or_none(r.get("as_of_date"))
                # construct instruction/input/output if missing
                instr = r.get("instruction") or "Given the market snapshot, produce reasoning, support and action."
                inp_parts = []
                if "indicators" in r:
                    for k, v in r["indicators"].items():
                        inp_parts.append(f"{k}={v}")
                input_text = " ; ".join(inp_parts) if inp_parts else ""
                output_text = ""
                # if reasoning/support/action exist, assemble
                if r.get("reasoning"):
                    output_text += f"<reasoning>{r['reasoning']}</reasoning>"
                if r.get("support"):
                    output_text += f"<support>{r['support']}</support>"
                if r.get("action"):
                    output_text += f"<action>{r['action']}</action>"
                # fallback
                r["instruction"] = instr
                r["input"] = input_text
                r["output"] = output_text
                # provenance
                r["source_file"] = os.path.basename(path)
                r["uid"] = f"{r.get('ticker') or 'UNK'}|{r.get('as_of_date') or 'UNK'}|{i}"
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1
    print(f"Wrote {total} records to {out_file}")

if __name__ == "__main__":
    main()
