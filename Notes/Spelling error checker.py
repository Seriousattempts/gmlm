import json
import time
from pathlib import Path
from collections import defaultdict


def extract_objects_from_file(file_path):
    """
    Extract JSON objects from a .jsonl file using robust parsing
    """
    try:
        raw = file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"    Error reading {file_path}: {e}")
        return []

    # Extract JSON objects using brace matching (same logic from previous code)
    objects = []
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
                        obj_str = raw[start:i + 1]
                        try:
                            obj = json.loads(obj_str)
                            objects.append(obj)
                        except json.JSONDecodeError:
                            pass
                        start = -1
        i += 1

    return objects


def get_nested_keys(data, parent_key=""):
    """
    Recursively get all keys from nested structures
    """
    keys = set()

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            keys.add(full_key)

            if isinstance(value, dict):
                keys.update(get_nested_keys(value, full_key))

    return keys


def get_all_unique_keys(folder_path):
    """
    Scan all .jsonl files and collect unique keys (including nested)
    """
    root_path = Path(folder_path)
    jsonl_files = list(root_path.rglob("*.jsonl"))

    unique_keys = set()
    files_scanned = 0

    print("🔍 Scanning files to find available query parameters...")

    for file_path in jsonl_files:
        objects = extract_objects_from_file(file_path)
        files_scanned += 1

        for obj in objects:
            if isinstance(obj, dict):
                unique_keys.update(get_nested_keys(obj))

        # Progress indicator
        if files_scanned % 10 == 0:
            print(f"   Scanned {files_scanned} files...")
            time.sleep(0.1)

    print(f"✅ Completed scanning {files_scanned} files")
    time.sleep(0.5)

    return sorted(unique_keys), jsonl_files


def get_nested_parameters(folder_path, parent_key):
    """
    Get unique parameters within a nested object
    """
    root_path = Path(folder_path)
    jsonl_files = list(root_path.rglob("*.jsonl"))

    nested_keys = set()

    print(f"🔍 Scanning for values in parameter '{parent_key}'...")

    for file_path in jsonl_files:
        objects = extract_objects_from_file(file_path)

        for obj in objects:
            nested_obj = get_nested_value(obj, parent_key)
            if isinstance(nested_obj, dict):
                nested_keys.update(nested_obj.keys())

    return sorted(nested_keys)


def get_nested_value(data, key_path):
    """
    Get value from nested structure using dot notation (e.g., 'text.Function')
    """
    keys = key_path.split('.')
    current = data

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None

    return current


def set_nested_value(data, key_path, new_value):
    """
    Set value in nested structure using dot notation
    """
    keys = key_path.split('.')
    current = data

    for key in keys[:-1]:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return False

    if isinstance(current, dict) and keys[-1] in current:
        current[keys[-1]] = new_value
        return True

    return False


def replace_in_nested_structure(data, target_key, search_text, replace_text):
    """
    Recursively search and replace text in nested JSON structures
    """
    modified = False

    # Handle dot notation for nested keys
    if '.' in target_key:
        value = get_nested_value(data, target_key)
        if isinstance(value, str) and search_text in value:
            new_value = value.replace(search_text, replace_text)
            if set_nested_value(data, target_key, new_value):
                modified = True
        return modified

    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                if isinstance(value, str) and search_text in value:
                    data[key] = value.replace(search_text, replace_text)
                    modified = True
            elif isinstance(value, (dict, list)):
                if replace_in_nested_structure(value, target_key, search_text, replace_text):
                    modified = True
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                if replace_in_nested_structure(item, target_key, search_text, replace_text):
                    modified = True

    return modified


def process_file_modifications(file_path, target_key, search_text, replace_text):
    """
    Process a single .jsonl file and apply modifications
    """
    objects = extract_objects_from_file(file_path)
    if not objects:
        return False, 0

    modifications_made = 0
    modified_objects = []

    for obj in objects:
        obj_modified = replace_in_nested_structure(obj, target_key, search_text, replace_text)
        if obj_modified:
            modifications_made += 1
        modified_objects.append(obj)

    # Write back to file if any modifications were made
    if modifications_made > 0:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for obj in modified_objects:
                    json_line = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
                    f.write(json_line + '\n')
            return True, modifications_made
        except Exception as e:
            print(f"    Error writing to {file_path}: {e}")
            return False, 0

    return False, 0


def spell_check_modifier(folder_path):
    """
    Main function to modify spelling in .jsonl files with nested parameter support
    """
    root_path = Path(folder_path)

    if not root_path.exists():
        print(f"❌ Error: Folder '{folder_path}' does not exist!")
        return

    print(f"🚀 Starting spell check modification for folder: {folder_path}")
    print("=" * 60)
    time.sleep(1)

    # Step 1: Get all unique keys (including nested)
    unique_keys, jsonl_files = get_all_unique_keys(folder_path)

    if not unique_keys:
        print("❌ No query parameters found in .jsonl files!")
        return

    # Filter to show only top-level keys first
    top_level_keys = [key for key in unique_keys if '.' not in key]

    # Step 2: Display available top-level query parameters
    print(f"\n📋 Found {len(top_level_keys)} unique query parameters:")
    print("-" * 40)
    for i, key in enumerate(top_level_keys, 1):
        print(f'   {i}. "{key}"')

    # Step 3: Get user selection for parameter to modify
    print("\n" + "=" * 60)
    while True:
        try:
            choice = input(f"👆 Select parameter to modify (1-{len(top_level_keys)}): ").strip()
            param_index = int(choice) - 1
            if 0 <= param_index < len(top_level_keys):
                selected_key = top_level_keys[param_index]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(top_level_keys)}")
        except ValueError:
            print("❌ Please enter a valid number")

    print(f'✅ Selected parameter: "{selected_key}"')
    time.sleep(0.5)

    # Step 4: Check if selected parameter has nested structure
    nested_params = get_nested_parameters(folder_path, selected_key)
    final_target_key = selected_key

    if nested_params:
        print(f"\n📋 Found {len(nested_params)} unique '{selected_key}' query parameters:")
        print("-" * 40)
        for i, param in enumerate(nested_params, 1):
            print(f'   {i}. "{param}"')

        print("\n" + "=" * 60)
        while True:
            try:
                choice = input(f"👆 Select parameter to modify (1-{len(nested_params)}): ").strip()
                param_index = int(choice) - 1
                if 0 <= param_index < len(nested_params):
                    nested_key = nested_params[param_index]
                    final_target_key = f"{selected_key}.{nested_key}"
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(nested_params)}")
            except ValueError:
                print("❌ Please enter a valid number")

        print(f'✅ Selected nested parameter: "{nested_key}"')
        time.sleep(0.5)

    # Step 5: Get search and replace terms
    print("\n" + "=" * 60)
    search_term = input("🔍 Enter text to search for: ").strip()
    if not search_term:
        print("❌ Search term cannot be empty!")
        return

    replace_term = input("✏️  Enter replacement text: ").strip()

    print(f'\n📝 Will replace "{search_term}" with "{replace_term}" in parameter "{final_target_key}"')

    # Step 6: Confirm operation
    confirmation = input("\n❓ Proceed with modifications? (y/n): ").strip().lower()
    if confirmation != 'y':
        print("❌ Operation cancelled.")
        return

    # Step 7: Process files
    print(f"\n🔧 Processing {len(jsonl_files)} files...")
    print("=" * 60)
    time.sleep(1)

    modified_files = []
    total_modifications = 0
    processed_files = 0

    for i, file_path in enumerate(jsonl_files, 1):
        print(f"📁 Processing file {i}/{len(jsonl_files)}: {file_path.name}")
        time.sleep(0.2)  # Small delay for visibility

        success, mod_count = process_file_modifications(file_path, final_target_key, search_term, replace_term)
        processed_files += 1

        if success:
            modified_files.append(file_path.name)
            total_modifications += mod_count
            print(f"   ✅ Modified {mod_count} entries")
        else:
            print(f"   ⏭️  No changes needed")

        # Progress update every 25 files
        if processed_files % 25 == 0:
            print(f"   📊 Progress: {processed_files}/{len(jsonl_files)} files processed")
            time.sleep(0.3)

    # Step 8: Summary
    print("\n" + "=" * 60)
    print("📊 MODIFICATION SUMMARY")
    print("=" * 60)
    print(f"🔢 Total files processed: {len(jsonl_files)}")
    print(f"✏️  Files modified: {len(modified_files)}")
    print(f"🔄 Total replacements made: {total_modifications}")
    print(f"🎯 Parameter modified: '{final_target_key}'")
    print(f"🔍 Search term: '{search_term}'")
    print(f"✨ Replace term: '{replace_term}'")

    if modified_files:
        print(f"\n📝 Modified .jsonl files:")
        print("-" * 30)
        for filename in modified_files:
            print(f"   • {filename}")
    else:
        print(f"\n💡 No files required modification.")

    print(f"\n🎉 Spell check modification completed!")
    time.sleep(1)


# Main execution
if __name__ == "__main__":
    print("🔤 JSONL Spell Check Modifier")
    print("=" * 60)

    # Get folder path from user
    folder_path = input("📁 Enter the folder path containing .jsonl files: ").strip()

    # Alternative: You can hardcode the path for testing
    # folder_path = r"C:\path\to\your\jsonl\files"

    print("\n" + "⏳" * 20)
    time.sleep(0.5)

    spell_check_modifier(folder_path)
