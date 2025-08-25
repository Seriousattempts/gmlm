import json
from pathlib import Path

def extract_objects(raw: str):
    """
    Yield candidate top-level JSON object strings by brace matching,
    ignoring braces inside strings and escapes.
    """
    objs = []
    n = len(raw)
    i = 0
    in_string = False
    escape = False
    depth = 0
    start = -1

    while i < n:
        ch = raw[i]

        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        # complete top-level object
                        objs.append(raw[start:i+1])
                        start = -1
        i += 1

    return objs

def trim_to_id_sheet_text(obj_str: str):
    """
    From a full object string, try to pull only id, sheet, and text as a well-formed object.
    Strategy:
      - Parse loosely by brace matching to find the text block boundaries.
      - Then build a minimal dict with id, sheet, text and return JSON string.
    If anything fails, return None.
    """
    # Quick sanity: must contain keys
    if '"id"' not in obj_str or '"sheet"' not in obj_str or '"text"' not in obj_str:
        return None

    # Try a tolerant parse by first removing outer noise via json.loads if possible.
    # If it fails (due to metadata tails), do a manual key extraction.
    data = None
    try:
        data = json.loads(obj_str)
    except Exception:
        # Manual extraction: find text object boundaries, then reconstruct
        try:
            # Find "id" value (simple heuristic: first "id" key)
            id_key = obj_str.find('"id"')
            if id_key == -1:
                return None
            # Find value after colon
            colon = obj_str.find(':', id_key)
            if colon == -1:
                return None
            # Extract string value (simple JSON string scan)
            def extract_json_string(s, start_idx):
                # skip whitespace
                j = start_idx
                while j < len(s) and s[j] in ' \t\r\n':
                    j += 1
                if j >= len(s) or s[j] != '"':
                    return None, j
                j += 1
                buf = []
                esc = False
                while j < len(s):
                    c = s[j]
                    if esc:
                        buf.append(c)
                        esc = False
                    elif c == '\\':
                        esc = True
                    elif c == '"':
                        return ''.join(buf), j+1
                    else:
                        buf.append(c)
                    j += 1
                return None, j

            id_val, _ = extract_json_string(obj_str, colon+1)
            if id_val is None:
                return None

            sheet_pos = obj_str.find('"sheet"', id_key)
            if sheet_pos == -1:
                return None
            sheet_colon = obj_str.find(':', sheet_pos)
            if sheet_colon == -1:
                return None
            sheet_val, _ = extract_json_string(obj_str, sheet_colon+1)
            if sheet_val is None:
                return None

            # Find "text": { ... } block boundaries by brace matching starting at the '{' after "text":
            text_key = obj_str.find('"text"', sheet_pos)
            if text_key == -1:
                return None
            after_text_colon = obj_str.find(':', text_key)
            if after_text_colon == -1:
                return None
            # Move to first '{'
            j = after_text_colon + 1
            while j < len(obj_str) and obj_str[j] not in '{':
                # tolerate whitespace
                if obj_str[j] in ' \t\r\n':
                    j += 1
                    continue
                # If not '{', text is malformed
                if obj_str[j] != '{':
                    return None
            if j >= len(obj_str) or obj_str[j] != '{':
                return None

            # Now brace-match the text object
            in_str = False
            esc = False
            depth = 0
            k = j
            while k < len(obj_str):
                c = obj_str[k]
                if in_str:
                    if esc:
                        esc = False
                    elif c == '\\':
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            # text object ends at k
                            text_block = obj_str[j:k+1]
                            # Try to parse text block
                            text_obj = json.loads(text_block)
                            return json.dumps({"id": id_val, "sheet": sheet_val, "text": text_obj}, ensure_ascii=False, separators=(',', ':'))
                k += 1
            return None
        except Exception:
            return None

    # If we could json.loads the whole object, construct the trimmed version cleanly
    try:
        if not isinstance(data, dict):
            return None
        if 'id' not in data or 'sheet' not in data or 'text' not in data:
            return None
        return json.dumps({
            "id": data['id'],
            "sheet": data['sheet'],
            "text": data['text']
        }, ensure_ascii=False, separators=(',', ':'))
    except Exception:
        return None

def process_jsonl_file(input_path: Path, save_dir: Path):
    """
    Read entire file as text, extract and trim objects, and write compact JSONL.
    """
    output_path = save_dir / f"{input_path.stem}_text{input_path.suffix}"

    try:
        raw = input_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"    Error reading {input_path}: {e}")
        return False, 0

    objects = extract_objects(raw)
    written = 0
    with open(output_path, 'w', encoding='utf-8') as out:
        for obj_str in objects:
            trimmed = trim_to_id_sheet_text(obj_str)
            if trimmed:
                out.write(trimmed + '\n')
                written += 1

    if written == 0:
        # optional: remove empty file
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False, 0

    return True, written

def process_folder(root_folder: str):
    root = Path(root_folder)
    if not root.exists():
        print(f"Error: Folder '{root_folder}' does not exist.")
        return

    # Create central _text directory as save_directory
    save_directory = root / "_text"
    save_directory.mkdir(exist_ok=True)

    # Gather all .jsonl files, excluding any under _text
    files = [p for p in root.rglob("*.jsonl") if save_directory not in p.parents]

    if not files:
        print("No .jsonl files found (excluding _text).")
        return

    total = len(files)
    print(f"Found {total} .jsonl file(s). Save directory: {save_directory}\n")

    ok = 0
    total_entries = 0

    for idx, f in enumerate(files, 1):
        remaining = total - idx
        print(f"# Processing file {idx}/{total}: {f.name}")
        print(f"# Remaining files: {remaining}")
        print(f"  Source: {f}")

        success, count = process_jsonl_file(f, save_directory)
        if success:
            ok += 1
            total_entries += count
            out_name = f"{f.stem}_text{f.suffix}"
            print(f"  ✓ Saved the new .jsonl file: {save_directory / out_name}")
            print(f"  ✓ Exported {count} objects")
        else:
            print("  ✗ No valid objects found (skipped)")
        print()

    print("=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Save directory: {save_directory}")
    print(f"Successfully processed: {ok}/{total} files")
    print(f"Total objects exported: {total_entries}")
    print(f"Failed files: {total - ok}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing .jsonl files: ").strip()
    print(f"\nStarting processing of folder: {folder_path}")
    print("-" * 50)
    process_folder(folder_path)
