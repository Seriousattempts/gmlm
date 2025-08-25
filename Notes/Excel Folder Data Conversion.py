import pandas as pd
import os
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import hashlib
import warnings
import sys
import re

# Handle optional dependencies gracefully
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars.")


    # Create a dummy tqdm class
    class tqdm:
        def __init__(self, iterable, desc="", *args, **kwargs):
            self.iterable = iterable
            self.desc = desc

        def __iter__(self):
            return iter(self.iterable)

        def set_description(self, desc):
            pass

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_output_folder_name(xlsx_filename: str) -> str:
    """Remove .xlsx extension and '_references' suffix to get folder name"""
    # Remove .xlsx extension
    base_name = xlsx_filename[:-5] if xlsx_filename.lower().endswith('.xlsx') else xlsx_filename
    # Remove '_references' suffix if present
    if base_name.endswith('_references'):
        base_name = base_name[:-11]
    return base_name


def clean_sheet_name_for_id(sheet_name: str) -> str:
    """Clean sheet name to be safe for use in IDs"""
    # Remove special characters and replace spaces with underscores
    cleaned = re.sub(r'[^\w\s-]', '', str(sheet_name))
    cleaned = re.sub(r'\s+', '_', cleaned)
    return cleaned[:50]  # Limit length


def clean_value_for_id(value: str) -> str:
    """Clean value to be safe for use in IDs"""
    cleaned = re.sub(r'[^\w\s-]', '', str(value))
    cleaned = re.sub(r'\s+', '_', cleaned)
    return cleaned[:30]  # Limit length


def is_labeled_column(col) -> bool:
    """Check if a column is properly labeled (not None, NaN, or empty)"""
    if col is None:
        return False
    if pd.isna(col):
        return False
    if isinstance(col, str):
        return bool(col.strip()) and not col.strip().lower().startswith('unnamed')
    # For numeric columns (like -1, 0, 1, 2), these are valid labels
    if isinstance(col, (int, float)):
        return not pd.isna(col)
    # Convert to string and check
    col_str = str(col).strip()
    return bool(col_str) and not col_str.lower().startswith('unnamed')


def title_case_column_name(col_name: str) -> str:
    """Convert column name to proper title case for display"""
    # Handle numeric column names
    if isinstance(col_name, (int, float)):
        return str(col_name)

    # Replace underscores and hyphens with spaces, then title case
    clean_name = str(col_name).replace('_', ' ').replace('-', ' ')
    return ' '.join(word.capitalize() for word in clean_name.split())


def normalize_column_for_pandas(col) -> str:
    """Normalize column name for pandas usecols parameter"""
    if isinstance(col, (int, float)):
        return str(int(col))  # Convert numeric columns to string integers
    return str(col).strip()


def should_preserve_formatting(col_name: str) -> bool:
    """Check if a column should preserve its original formatting (line breaks, etc.)"""
    col_lower = str(col_name).lower()
    # Preserve formatting for code-related columns
    preserve_keywords = ['sample', 'code', 'snippet', 'example', 'syntax', 'format', 'script', 'program']
    return any(keyword in col_lower for keyword in preserve_keywords)


class ExcelToJsonlConverter:
    def __init__(self,
                 folder_path: str,
                 max_file_size_mb: float = 100.0):
        """
        Initialize the Excel to JSONL Converter

        Args:
            folder_path: Path to folder containing Excel files
            max_file_size_mb: Maximum file size to process (MB)
        """
        self.folder_path = Path(folder_path)

        # Validate folder path
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder path does not exist: {folder_path}")
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {folder_path}")

        self.max_file_size_mb = max_file_size_mb

        # Processing stats
        self.processing_stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'total_records': 0,
            'total_sheets': 0,
            'total_unlabeled_columns_filtered': 0,
            'errors': [],
            'warnings': [],
            'processing_time': 0
        }

    def get_excel_files(self) -> List[Path]:
        """Get all Excel files from the specified folder"""
        excel_extensions = ['.xlsx', '.xls', '.xlsm', '.xlsb']
        excel_files = []

        try:
            # Search for Excel files
            for ext in excel_extensions:
                excel_files.extend(self.folder_path.glob(f"*{ext}"))
                excel_files.extend(self.folder_path.glob(f"*{ext.upper()}"))

            # Remove duplicates and sort
            excel_files = sorted(list(set(excel_files)))

            # Filter out temporary files and hidden files
            excel_files = [f for f in excel_files
                           if not f.name.startswith('~$') and not f.name.startswith('.')]

            logger.info(f"Found {len(excel_files)} Excel files")
            return excel_files

        except Exception as e:
            logger.error(f"Error scanning for Excel files: {e}")
            return []

    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        """Enhanced file validation"""
        try:
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return False, f"File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB"

            # Check if file is accessible
            if not os.access(file_path, os.R_OK):
                return False, "File is not readable (permission denied)"

            # Try to open the file to check if it's corrupted
            try:
                with pd.ExcelFile(file_path, engine='openpyxl') as excel_file:
                    if not excel_file.sheet_names:
                        return False, "No sheets found in file"
            except Exception as e:
                try:
                    with pd.ExcelFile(file_path, engine='xlrd') as excel_file:
                        if not excel_file.sheet_names:
                            return False, "No sheets found in file"
                except Exception:
                    return False, f"Cannot read Excel file: {str(e)}"

            return True, ""

        except Exception as e:
            return False, f"File validation failed: {str(e)}"

    def identify_all_sheets_and_columns(self, file_path: Path) -> Dict[str, List[str]]:
        """
        First identify all sheets and their columns, then filter unlabeled columns
        Returns: {sheet_name: [labeled_column_names]}
        """
        sheet_columns = {}
        total_unlabeled = 0

        try:
            # Load Excel file and get all sheet names first
            with pd.ExcelFile(file_path, engine='openpyxl') as excel_file:
                all_sheet_names = excel_file.sheet_names

                logger.info(f"  Identified {len(all_sheet_names)} sheets in {file_path.name}")

                # Now process each sheet and filter unlabeled columns
                for sheet_name in all_sheet_names:
                    try:
                        # Read only the header row to get column names
                        df_header = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=0)
                        all_columns = df_header.columns.tolist()

                        # Filter out unlabeled columns (including handling numeric columns)
                        labeled_columns = []
                        unlabeled_count = 0

                        for col in all_columns:
                            if is_labeled_column(col):
                                # Normalize column name for pandas
                                normalized_col = normalize_column_for_pandas(col)
                                labeled_columns.append(normalized_col)
                            else:
                                unlabeled_count += 1

                        sheet_columns[sheet_name] = labeled_columns
                        total_unlabeled += unlabeled_count

                        if unlabeled_count > 0:
                            logger.info(f"    Sheet '{sheet_name}': {len(labeled_columns)} labeled columns, "
                                        f"{unlabeled_count} unlabeled columns filtered out")
                        else:
                            logger.info(f"    Sheet '{sheet_name}': {len(labeled_columns)} labeled columns")

                    except Exception as e:
                        logger.warning(f"    Error reading sheet '{sheet_name}': {e}")
                        sheet_columns[sheet_name] = []
                        continue

        except Exception as e:
            logger.error(f"Error identifying sheets and columns in {file_path.name}: {e}")
            return {}

        self.processing_stats['total_unlabeled_columns_filtered'] += total_unlabeled

        if total_unlabeled > 0:
            logger.info(f"  Total unlabeled columns filtered from {file_path.name}: {total_unlabeled}")

        return sheet_columns

    def read_sheet_with_numeric_columns(self, file_path: Path, sheet_name: str,
                                        labeled_columns: List[str]) -> pd.DataFrame:
        """Read sheet data handling numeric column names properly"""
        try:
            # First approach: Try reading with the normalized column names
            try:
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    usecols=labeled_columns,
                    engine='openpyxl',
                    keep_default_na=True,
                    na_values=['', 'N/A', 'NULL', 'null', 'None', '#N/A', '#NULL!']
                )
                logger.debug(f"Successfully read sheet '{sheet_name}' with specified columns")
            except Exception as e:
                logger.warning(f"Failed to read with usecols, trying fallback: {e}")
                # Fallback: Read entire sheet and filter columns afterwards
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    engine='openpyxl',
                    keep_default_na=True,
                    na_values=['', 'N/A', 'NULL', 'null', 'None', '#N/A', '#NULL!']
                )

                # Filter to only the columns we want
                available_cols = []
                for col in df.columns:
                    normalized_col = normalize_column_for_pandas(col)
                    if normalized_col in labeled_columns:
                        available_cols.append(col)

                if available_cols:
                    df = df[available_cols]
                    logger.info(f"Filtered to {len(available_cols)} available columns in '{sheet_name}'")
                else:
                    logger.warning(f"No matching columns found in sheet '{sheet_name}'")
                    return pd.DataFrame()

            if df.empty:
                logger.info(f"Sheet '{sheet_name}' is empty")
                return df

            # Ensure column names are clean strings
            df.columns = [str(col).strip() for col in df.columns]

            # Remove completely empty rows
            df = df.dropna(how='all')

            logger.debug(f"Read {len(df)} rows from sheet '{sheet_name}'")
            return df

        except Exception as e:
            error_msg = f"Error reading sheet '{sheet_name}' from {file_path.name}: {e}"
            logger.error(error_msg)
            self.processing_stats['errors'].append(error_msg)
            return pd.DataFrame()

    def clean_text(self, text: Union[str, Any], col_name: str = "") -> str:
        """Enhanced text cleaning with formatting preservation option"""
        if pd.isna(text) or text is None:
            return ""

        try:
            text = str(text).strip()

            # Remove Excel formula artifacts
            if text.startswith('='):
                return ""

            # Handle Excel error values
            excel_errors = ['#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#NUM!', '#NULL!', '#N/A']
            if text in excel_errors:
                return ""

            # Check if we should preserve formatting for this column
            if should_preserve_formatting(col_name):
                # Preserve line breaks and formatting for code-related columns
                # Only normalize excessive whitespace within lines
                lines = text.split('\n')
                cleaned_lines = []
                for line in lines:
                    # Clean each line individually but preserve the line structure
                    cleaned_line = re.sub(r'[ \t]+', ' ', line.strip())
                    if cleaned_line:  # Only add non-empty lines
                        cleaned_lines.append(cleaned_line)

                # Rejoin with newlines
                return '\n'.join(cleaned_lines)
            else:
                # Standard text cleaning for non-code columns
                # Apply lowercase only to description-like columns
                if any(desc_word in col_name.lower() for desc_word in ['description', 'desc']):
                    text = text.lower()

                # Standard whitespace normalization
                text = re.sub(r'\s+', ' ', text)
                text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

                return text.strip()

        except Exception as e:
            logger.debug(f"Error cleaning text '{str(text)[:50]}...': {e}")
            return ""

    def generate_file_id(self, file_path: Path) -> str:
        """Generate ID from filename (without _references.xlsx)"""
        return get_output_folder_name(file_path.name)

    def process_single_file(self, file_path: Path) -> int:
        """Process a single Excel file and create JSONL output"""
        logger.info(f"Processing file: {file_path.name}")

        # Validate file
        is_valid, error_msg = self.validate_file(file_path)
        if not is_valid:
            logger.error(f"Skipping {file_path.name}: {error_msg}")
            self.processing_stats['files_skipped'] += 1
            self.processing_stats['warnings'].append(f"{file_path.name}: {error_msg}")
            return 0

        try:
            # STEP 1: First identify ALL sheets and filter unlabeled columns
            sheet_columns_map = self.identify_all_sheets_and_columns(file_path)

            if not sheet_columns_map:
                logger.warning(f"No valid sheets found in {file_path.name}")
                self.processing_stats['files_skipped'] += 1
                return 0

            all_records = []
            sheets_processed = 0

            # Get file ID
            file_id = self.generate_file_id(file_path)

            # STEP 2: Now process each sheet using only the labeled columns
            for sheet_name, labeled_columns in sheet_columns_map.items():
                if not labeled_columns:
                    logger.info(f"    Skipping sheet '{sheet_name}' - no labeled columns")
                    continue

                try:
                    logger.info(f"  Processing sheet: {sheet_name} with {len(labeled_columns)} labeled columns")

                    # Read sheet data using the special method for numeric columns
                    df = self.read_sheet_with_numeric_columns(file_path, sheet_name, labeled_columns)

                    if df.empty:
                        logger.info(f"    Sheet '{sheet_name}' has no data after reading - skipping")
                        continue

                    # Process each row
                    row_count = 0
                    for idx, row in df.iterrows():
                        try:
                            # Skip rows where all values are NaN
                            if row.isna().all():
                                continue

                            # Build text object - combine all labeled columns as JSON object
                            text_object = {}
                            non_empty_columns = []

                            for col in df.columns:
                                try:
                                    # Use column name for formatting decision
                                    value = self.clean_text(row[col], col_name=col)
                                    if value:  # Only include non-empty values
                                        # Create proper title case key
                                        display_col_name = title_case_column_name(col)
                                        text_object[display_col_name] = value
                                        non_empty_columns.append(str(col))
                                except Exception as e:
                                    logger.debug(f"Error processing column '{col}' in row {idx}: {e}")
                                    continue

                            # Skip if no meaningful content
                            if not text_object:
                                continue

                            # New record structure with separate id and sheet
                            record = {
                                'id': file_id,
                                'sheet': sheet_name,
                                'text': text_object,  # JSON object with preserved formatting
                                'metadata': {
                                    'file_name': file_path.name,
                                    'sheet_name': sheet_name,
                                    'row_index': int(idx),
                                    'total_labeled_columns': len(labeled_columns),
                                    'non_empty_columns': len(non_empty_columns),
                                    'columns_used': non_empty_columns,
                                    'text_length': sum(len(str(v)) for v in text_object.values()),
                                    'word_count': sum(len(str(v).split()) for v in text_object.values()),
                                    'processing_timestamp': datetime.now().isoformat()
                                }
                            }

                            all_records.append(record)
                            row_count += 1

                        except Exception as e:
                            logger.debug(f"Error processing row {idx} in sheet '{sheet_name}': {e}")
                            continue

                    if row_count > 0:
                        logger.info(f"    Processed {row_count} rows from sheet '{sheet_name}'")
                        sheets_processed += 1
                    else:
                        logger.info(f"    No valid rows found in sheet '{sheet_name}'")

                except Exception as e:
                    error_msg = f"Error processing sheet '{sheet_name}' in {file_path.name}: {e}"
                    logger.error(error_msg)
                    self.processing_stats['errors'].append(error_msg)
                    continue

            # Only create output if we have records
            if not all_records:
                logger.warning(f"No records created for file {file_path.name}")
                self.processing_stats['files_skipped'] += 1
                return 0

            # Create output folder
            folder_name = get_output_folder_name(file_path.name)
            output_folder = self.folder_path / folder_name

            try:
                output_folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                error_msg = f"Could not create output folder {output_folder}: {e}"
                logger.error(error_msg)
                self.processing_stats['errors'].append(error_msg)
                self.processing_stats['files_skipped'] += 1
                return 0

            # Create output file
            output_file = output_folder / f"{folder_name}.jsonl"

            # Write JSONL with pretty printing
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for record in all_records:
                        json_line = json.dumps(record, ensure_ascii=False, indent=2)
                        f.write(json_line + '\n')

                logger.info(f"File {file_path.name} complete: {len(all_records)} records from "
                            f"{sheets_processed} sheets written to {output_file}")

                self.processing_stats['files_processed'] += 1
                self.processing_stats['total_sheets'] += sheets_processed

                return len(all_records)

            except Exception as e:
                error_msg = f"Error writing output file {output_file}: {e}"
                logger.error(error_msg)
                self.processing_stats['errors'].append(error_msg)
                self.processing_stats['files_skipped'] += 1
                return 0

        except Exception as e:
            error_msg = f"Error processing file {file_path.name}: {e}"
            logger.error(error_msg)
            self.processing_stats['errors'].append(error_msg)
            self.processing_stats['files_skipped'] += 1
            return 0

    def process_all_files(self, show_progress: bool = True) -> int:
        """Process all Excel files in the folder"""
        start_time = datetime.now()
        excel_files = self.get_excel_files()

        if not excel_files:
            logger.warning("No Excel files found in the specified folder")
            return 0

        total_records = 0

        # Setup progress tracking
        if show_progress and HAS_TQDM:
            file_progress = tqdm(excel_files, desc="Processing Excel files")
        else:
            file_progress = excel_files

        for file_path in file_progress:
            if show_progress and HAS_TQDM:
                file_progress.set_description(f"Processing {file_path.name}")

            records_count = self.process_single_file(file_path)
            total_records += records_count

        # Calculate processing time
        end_time = datetime.now()
        self.processing_stats['processing_time'] = (end_time - start_time).total_seconds()
        self.processing_stats['total_records'] = total_records

        logger.info(f"Processing complete: {total_records} total records from "
                    f"{self.processing_stats['files_processed']} files in "
                    f"{self.processing_stats['processing_time']:.2f} seconds")

        return total_records

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary"""
        files_found = self.processing_stats['files_processed'] + self.processing_stats['files_skipped']

        return {
            'processing_stats': self.processing_stats,
            'summary': {
                'files_found': files_found,
                'files_processed': self.processing_stats['files_processed'],
                'files_skipped': self.processing_stats['files_skipped'],
                'total_sheets_processed': self.processing_stats['total_sheets'],
                'total_records_created': self.processing_stats['total_records'],
                'total_unlabeled_columns_filtered': self.processing_stats['total_unlabeled_columns_filtered'],
                'processing_time_seconds': self.processing_stats['processing_time'],
                'error_count': len(self.processing_stats['errors']),
                'warning_count': len(self.processing_stats['warnings']),
                'average_records_per_file': (self.processing_stats['total_records'] /
                                             max(1, self.processing_stats['files_processed'])),
                'processing_rate_files_per_second': (self.processing_stats['files_processed'] /
                                                     max(1, self.processing_stats['processing_time']))
            }
        }


def main():
    """Main function"""

    print("🚀 Excel to JSONL Converter - One JSONL per Excel File")
    print("=" * 60)
    print("This tool processes each Excel file individually and creates:")
    print("• A folder named after each Excel file (removing '_references' suffix)")
    print("• One JSONL file per Excel file containing data from ALL sheets")
    print("• Handles numeric column names (including negative numbers)")
    print("• Preserves line breaks and formatting for code samples")
    print("• Filters out unlabeled columns from all sheets")
    print("• Uses only labeled columns for data extraction")
    print("• Creates structured IDs with separate file and sheet fields")
    print("• Formats text as JSON objects with proper key-value pairs")
    print()

    # Get folder path from user
    folder_path = input("Enter the path to the folder containing .xlsx files: ").strip()

    # Remove quotes if user included them
    folder_path = folder_path.strip('"\'')

    # Validate path
    if not folder_path:
        print("❌ Error: No path provided")
        return None

    if not os.path.exists(folder_path):
        print(f"❌ Error: Path does not exist: {folder_path}")
        return None

    try:
        # Initialize converter
        converter = ExcelToJsonlConverter(
            folder_path=folder_path,
            max_file_size_mb=100.0
        )

        print(f"📁 Processing folder: {folder_path}")
        print()

        # Process all files
        total_records = converter.process_all_files(show_progress=True)

        if total_records > 0:
            # Get summary
            summary = converter.get_processing_summary()

            # Display results
            print("\n" + "=" * 80)
            print("🎉 EXCEL TO JSONL CONVERSION COMPLETE")
            print("=" * 80)

            print(f"✅ Successfully processed {summary['summary']['files_processed']} Excel files")
            print(f"📊 Total records created: {summary['summary']['total_records_created']:,}")
            print(f"📄 Total sheets processed: {summary['summary']['total_sheets_processed']}")
            print(f"🚫 Total unlabeled columns filtered: {summary['summary']['total_unlabeled_columns_filtered']}")
            print(f"⏱️  Processing time: {summary['summary']['processing_time_seconds']:.2f} seconds")
            print(f"📈 Average records per file: {summary['summary']['average_records_per_file']:.1f}")
            print(f"⚡ Processing rate: {summary['summary']['processing_rate_files_per_second']:.2f} files/second")

            if summary['summary']['files_skipped'] > 0:
                print(f"⚠️  Files skipped: {summary['summary']['files_skipped']}")

            if summary['summary']['error_count'] > 0:
                print(f"❌ Errors encountered: {summary['summary']['error_count']}")
                print("Specific errors:")
                for error in summary['processing_stats']['errors'][:10]:  # Show first 10 errors
                    print(f"   • {error}")
                if len(summary['processing_stats']['errors']) > 10:
                    print(f"   ... and {len(summary['processing_stats']['errors']) - 10} more errors")

            if summary['summary']['warning_count'] > 0:
                print(f"⚠️  Warnings: {summary['summary']['warning_count']}")
                print("Specific warnings:")
                for warning in summary['processing_stats']['warnings'][:10]:  # Show first 10 warnings
                    print(f"   • {warning}")
                if len(summary['processing_stats']['warnings']) > 10:
                    print(f"   ... and {len(summary['processing_stats']['warnings']) - 10} more warnings")

            print(f"\n📋 Output Structure:")
            print(f"   • Each Excel file creates its own folder")
            print(f"   • Folder names remove '_references' suffix")
            print(f"   • Each folder contains one JSONL file with all data from that Excel file")
            print(f"   • Handles numeric columns (including negative numbers like -1, 0, 1, 2)")
            print(f"   • Line breaks preserved for code samples")
            print(f"   • Only labeled columns are processed (unlabeled columns filtered out)")
            print(f"   • IDs structured as separate file and sheet fields")
            print(f"   • Text formatted as JSON objects with proper key-value pairs")
            print(f"   • JSONL files are pretty-printed for readability")

            print(f"\n📋 Next steps:")
            print(f"   1. Check the created folders in: {folder_path}")
            print(f"   2. Each JSONL file contains properly formatted JSON objects")
            print(f"   3. Use these files for embedding generation with sentence-transformers")

            return summary
        else:
            print("❌ No data was processed.")
            print("   • Check if Excel files contain readable data with labeled columns")
            print("   • Verify file permissions")
            print("   • Check the log messages above for specific errors")
            return None

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    main()
