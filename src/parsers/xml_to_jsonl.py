#!/usr/bin/env python3
"""
xml_to_jsonl.py
Parse a directory of XML files and emit one JSON object per <thesis>.
Saves NDJSON (JSONL) to an output file with validation.

Usage:
  python src/parsers/xml_to_jsonl.py --input_dir data/raw_xml --output_file data/jsonl/all.jsonl
"""
import argparse
import os
import sys
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.validation import check_xml_structure, normalize_date, normalize_action
from src.utils.logger import setup_logger

# Load environment
load_dotenv()

logger = setup_logger(__name__)


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
    Given an Element for <thesis>, return dict with keys reasoning, support, action, as_of_date, and any indicators.
    """
    data = {}
    # try to pull standard tags including as-of-date
    for tag in ("reasoning", "support", "action"):
        node = elem.find(tag)
        if node is not None and node.text is not None:
            data[tag] = node.text.strip()
        else:
            data[tag] = None
    
    # Extract as-of-date specifically (handle both as-of-date and as_of_date)
    as_of_node = elem.find("as-of-date")
    if as_of_node is None:
        as_of_node = elem.find("as_of_date")
    if as_of_node is not None and as_of_node.text is not None:
        data["as_of_date"] = normalize_value(as_of_node.text)
    else:
        data["as_of_date"] = None
    
    # capture any other child tags as indicators / features
    indicators = {}
    for child in elem:
        tag = child.tag.lower()
        # Skip already processed tags
        if tag in ("reasoning", "support", "action", "as-of-date", "as_of_date"):
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
        logger.error(f"Failed to parse {path}: {e}")
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
            # attempt to find ticker/as-of-date on parent or self (as attributes)
            ticker = th.get("ticker") or root.get("ticker") or None
            rec["ticker"] = normalize_value(ticker) if ticker else rec.get("ticker")
            # Only override as_of_date if not already set by parse_thesis
            if not rec.get("as_of_date"):
                as_of = th.get("as-of-date") or th.get("as_of_date") or None
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
            # as-of-date can be on thesis or on parent (as attribute) - only override if not already set
            if not rec.get("as_of_date"):
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
    parser.add_argument("--input_dir", default=None, help="Directory with XML files (default from .env)")
    parser.add_argument("--output_file", default=None, help="Output NDJSON (jsonl) (default from .env)")
    parser.add_argument("--validate", action="store_true", default=True, help="Validate XML structure")
    args = parser.parse_args()

    # Get defaults from environment
    input_dir = args.input_dir or os.getenv("RAW_XML_DIR", "data/raw_xml")
    out_file = args.output_file or os.getenv("JSONL_OUTPUT", "data/jsonl/all.jsonl")
    
    logger.info("=" * 80)
    logger.info("Converting XML to JSONL")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output file: {out_file}")
    
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".xml") or f.lower().endswith(".txt")]
    logger.info(f"Found {len(files)} XML files")
    
    total = 0
    errors = 0
    
    with open(out_file, "w", encoding="utf-8") as fo:
        for path in files:
            logger.info(f"Processing: {os.path.basename(path)}")
            
            # Validate XML structure if requested
            if args.validate:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        xml_content = f.read()
                    is_valid, xml_errors = check_xml_structure(xml_content)
                    if not is_valid:
                        logger.warning(f"XML validation failed for {path}: {xml_errors}")
                except Exception as e:
                    logger.error(f"Cannot read {path}: {e}")
                    errors += 1
                    continue
            
            recs = parse_file(path)
            for i, r in enumerate(recs):
                # ensure minimal fields
                r["ticker"] = r.get("ticker") or None
                r["as_of_date"] = normalize_date(r.get("as_of_date")) or iso_date_or_none(r.get("as_of_date"))
                
                # Normalize action
                if r.get("action"):
                    r["action"] = normalize_action(r["action"]) or r["action"]
                
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
    
    logger.info("=" * 80)
    logger.info(f"Conversion complete!")
    logger.info(f"Wrote {total} records to {out_file}")
    logger.info(f"Errors: {errors}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
