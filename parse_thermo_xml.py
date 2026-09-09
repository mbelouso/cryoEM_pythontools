#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def _strip_ns(tag):
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_typed_value(value_elem):
    if value_elem is None:
        return None
    nil_attr = value_elem.attrib.get("{http://www.w3.org/2001/XMLSchema-instance}nil")
    if nil_attr == "true":
        return None
    raw_text = value_elem.text.strip() if value_elem.text else ""
    type_attr = value_elem.attrib.get("{http://www.w3.org/2001/XMLSchema-instance}type", "")
    type_name = type_attr.split(":", 1)[-1] if ":" in type_attr else type_attr

    if type_name in {"double", "float"}:
        try:
            return float(raw_text)
        except ValueError:
            return raw_text
    if type_name in {"int", "long", "short"}:
        try:
            return int(raw_text)
        except ValueError:
            return raw_text
    if type_name == "boolean":
        return raw_text.lower() == "true"

    return raw_text


def _parse_custom_data(root, namespaces):
    custom_data = {}
    custom_elem = root.find("m:CustomData", namespaces)
    if custom_elem is None:
        return custom_data

    for kv in custom_elem.findall("a:KeyValueOfstringanyType", namespaces):
        key_elem = kv.find("a:Key", namespaces)
        value_elem = kv.find("a:Value", namespaces)
        if key_elem is None:
            continue
        key = key_elem.text.strip() if key_elem.text else ""
        if not key:
            continue
        custom_data[key] = _parse_typed_value(value_elem)

    return custom_data


def _flatten_leaf_text(elem, path, out):
    nil_attr = elem.attrib.get("{http://www.w3.org/2001/XMLSchema-instance}nil")
    if nil_attr == "true":
        out["/".join(path)] = None
        return

    children = list(elem)
    if not children:
        text = elem.text.strip() if elem.text else ""
        if text != "":
            out["/".join(path)] = text
        return

    for child in children:
        _flatten_leaf_text(child, path + [_strip_ns(child.tag)], out)


def parse_microscope_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    namespaces = {
        "m": "http://schemas.datacontract.org/2004/07/Fei.SharedObjects",
        "a": "http://schemas.microsoft.com/2003/10/Serialization/Arrays",
        "i": "http://www.w3.org/2001/XMLSchema-instance",
    }

    data = {
        "customData": _parse_custom_data(root, namespaces),
        "fields": {},
    }

    _flatten_leaf_text(root, [_strip_ns(root.tag)], data["fields"])

    return data


def main():
    parser = argparse.ArgumentParser(description="Parse Thermo/FEI microscope XML metadata.")
    parser.add_argument("xml_path", type=Path, help="Path to the microscope XML file")
    parser.add_argument("-o", "--output", type=Path, help="Optional output JSON file")
    args = parser.parse_args()

    data = parse_microscope_xml(args.xml_path)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
    else:
        print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
