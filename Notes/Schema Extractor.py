import json
from pathlib import Path
from typing import Set, List, Dict, Any
import sys


class JSONLSchemaExtractor:
    """Single File Schema Extractor for JSONL files"""

    def __init__(self):
        self.stats = {
            'files_processed': 0,
            'total_records': 0,
            'failed_records': 0
        }

    def extract_text_fields_from_jsonl(self, jsonl_file_path: str) -> Set[str]:
        """
        Extract all unique text fields from a JSONL file's 'text' objects.
        Handles both standard JSONL and concatenated JSON objects.
        """
        text_fields = set()
        file_path = Path(jsonl_file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file_path}")

        print(f"📖 Analyzing: {file_path.name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # Handle both line-by-line JSONL and concatenated JSON objects
            records = self._parse_jsonl_content(content)

            for record in records:
                if isinstance(record, dict) and 'text' in record:
                    text_obj = record['text']
                    if isinstance(text_obj, dict):
                        text_fields.update(text_obj.keys())
                        self.stats['total_records'] += 1
                    else:
                        self.stats['failed_records'] += 1
                else:
                    self.stats['failed_records'] += 1

            print(f"✅ Found {len(text_fields)} unique text fields from {self.stats['total_records']} records")
            return text_fields

        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            return set()

    def _parse_jsonl_content(self, content: str) -> List[Dict[str, Any]]:
        """Parse JSONL content, handling both standard and concatenated formats"""
        records = []

        # Try standard JSONL first (line by line)
        if '\n' in content:
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    # If line-by-line fails, fall back to concatenated parsing
                    break

        # If standard JSONL failed or no newlines, try concatenated JSON parsing
        if not records:
            records = self._parse_concatenated_json(content)

        return records

    def _parse_concatenated_json(self, content: str) -> List[Dict[str, Any]]:
        """Parse concatenated JSON objects using brace matching"""
        records = []
        current_object = ""
        brace_depth = 0
        in_string = False
        escape_next = False

        for char in content:
            current_object += char

            if char == '"' and not escape_next:
                in_string = not in_string
            elif char == '\\' and in_string:
                escape_next = not escape_next
                continue

            if not in_string:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1

                    if brace_depth == 0:
                        obj_str = current_object.strip()
                        if obj_str:
                            try:
                                record = json.loads(obj_str)
                                records.append(record)
                            except json.JSONDecodeError:
                                pass
                        current_object = ""

            escape_next = False

        return records

    def create_schema_file(self, jsonl_file_path: str) -> str:
        """
        Create a _texted.txt schema file for the given JSONL file.
        Returns the path to the created schema file.
        """
        file_path = Path(jsonl_file_path)

        # Extract text fields
        text_fields = self.extract_text_fields_from_jsonl(jsonl_file_path)

        if not text_fields:
            print(f"⚠️ No text fields found in {file_path.name}")
            return ""

        # Create schema data
        schema_data = {
            "text": {
                "text_fields": sorted(list(text_fields))
            }
        }

        # Create output file path
        output_file_path = file_path.with_name(f"{file_path.stem}_texted.txt")

        # Write schema file
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(schema_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Schema saved to: {output_file_path}")
            print(f"🔍 Text fields discovered: {schema_data['text']['text_fields']}")

            self.stats['files_processed'] += 1
            return str(output_file_path)

        except Exception as e:
            print(f"❌ Failed to write schema file: {e}")
            return ""

    def process_folder(self, folder_path: str) -> None:
        """Process all JSONL files in a folder and create schema files for each"""
        folder = Path(folder_path)

        if not folder.exists():
            print(f"❌ Folder not found: {folder_path}")
            return

        jsonl_files = list(folder.glob("*.jsonl"))

        if not jsonl_files:
            print(f"❌ No JSONL files found in: {folder_path}")
            return

        print(f"📁 Found {len(jsonl_files)} JSONL files in {folder.name}")
        print("-" * 50)

        for jsonl_file in jsonl_files:
            self.create_schema_file(str(jsonl_file))
            print()

        self._print_summary()

    def _print_summary(self) -> None:
        """Print processing summary"""
        print("=" * 50)
        print("SCHEMA EXTRACTION COMPLETE")
        print("=" * 50)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Total records analyzed: {self.stats['total_records']}")
        print(f"Failed records: {self.stats['failed_records']}")


def main():
    """Main function for interactive use"""
    print("🔍 JSONL Schema Extractor - Single File Version")
    print("=" * 60)
    print("Creates {filename}_texted.txt files with discovered text fields")
    print("=" * 60)

    extractor = JSONLSchemaExtractor()

    try:
        choice = input("\nProcess (1) Single file or (2) Folder? Enter 1 or 2: ").strip()

        if choice == "1":
            file_path = input("📄 Enter path to JSONL file: ").strip().strip('"\'')
            if file_path:
                extractor.create_schema_file(file_path)
            else:
                print("❌ No file path provided")

        elif choice == "2":
            folder_path = input("📁 Enter path to folder containing JSONL files: ").strip().strip('"\'')
            if folder_path:
                extractor.process_folder(folder_path)
            else:
                print("❌ No folder path provided")

        else:
            print("❌ Invalid choice. Please enter 1 or 2.")

    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
