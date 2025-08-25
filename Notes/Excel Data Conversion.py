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
from collections import defaultdict

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


def count_sheet_column_names(folder_path: str) -> Dict:
    """
    Analyzes all .xlsx files in a folder and counts how many sheets
    share exactly the same column names, including file names.

    Args:
        folder_path (str): Path to the folder containing Excel files

    Returns:
        Dict: Analysis results with column patterns and their occurrences
    """
    # Dictionary to store column name patterns and their occurrences
    column_name_patterns = defaultdict(list)

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return {}

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx'):
            filepath = os.path.join(folder_path, filename)
            try:
                # Load the Excel file
                xls = pd.ExcelFile(filepath)

                # Check each sheet
                for sheet_name in xls.sheet_names:
                    try:
                        # Read only the header row to get column names
                        df = pd.read_excel(xls, sheet_name=sheet_name, nrows=0)

                        # Filter to only labeled columns for pattern matching
                        labeled_columns = [col for col in df.columns if is_labeled_column(col)]

                        # Get columns as tuple (immutable and hashable)
                        columns = tuple(labeled_columns)

                        # Store file name and sheet name for this column pattern
                        column_name_patterns[columns].append({
                            'file': filename,
                            'sheet': sheet_name
                        })

                    except Exception as e:
                        print(f"Warning: Could not read sheet '{sheet_name}' in '{filename}': {e}")

            except Exception as e:
                print(f"Warning: Could not read file '{filename}': {e}")

    # Check if any Excel files were found
    if not column_name_patterns:
        print("No Excel files with readable sheets found in the specified folder.")
        return {}

    # Sort the column patterns by their count from greatest to least
    sorted_patterns = sorted(column_name_patterns.items(),
                             key=lambda x: len(x[1]), reverse=True)

    # Print the results
    print("Sheet Column Name Analysis Results:")
    print("=" * 60)

    analysis_results = {}

    for i, (columns, occurrences) in enumerate(sorted_patterns, 1):
        count = len(occurrences)
        print(f"\nRank {i}: {count} sheets share these columns:")

        # Print column names
        if len(columns) > 0:
            print("Columns:")
            for col in columns:
                print(f"  - {col}")
        else:
            print("Columns: (No columns/empty sheet)")

        print(f"\nFound in these files and sheets:")
        # Group by file name for cleaner output
        files_dict = defaultdict(list)
        for occurrence in occurrences:
            files_dict[occurrence['file']].append(occurrence['sheet'])

        for file_name, sheet_names in files_dict.items():
            print(f"  📁 {file_name}")
            for sheet_name in sheet_names:
                print(f"    └── {sheet_name}")

        print("-" * 60)

        # Store in results for programmatic access
        analysis_results[i] = {
            'columns': list(columns),
            'count': count,
            'occurrences': occurrences
        }

    return analysis_results


class ExcelDataConverter:
    def __init__(self,
                 folder_path: str,
                 target_columns: List[str],
                 target_sheets: List[Dict[str, str]],
                 output_file: str,
                 chunk_size: int = 1000,
                 max_file_size_mb: float = 100.0):
        """
        Initialize the Excel Data Converter for exact column matching

        Args:
            folder_path: Path to folder containing Excel files
            target_columns: Exact columns that must match
            target_sheets: List of dicts with 'file' and 'sheet' keys for target sheets
            output_file: Output file name for processed data
            chunk_size: Number of rows to process at once for memory efficiency
            max_file_size_mb: Maximum file size to process (MB)
        """
        self.folder_path = Path(folder_path)

        # Validate folder path
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder path does not exist: {folder_path}")
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {folder_path}")

        self.target_columns = target_columns
        self.target_sheets = target_sheets
        self.output_file = output_file
        self.chunk_size = chunk_size
        self.max_file_size_mb = max_file_size_mb
        self.processed_data = []

        # Enhanced processing stats
        self.processing_stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'sheets_processed': 0,
            'sheets_skipped': 0,
            'target_sheets_found': 0,
            'rows_processed': 0,
            'rows_skipped': 0,
            'total_unlabeled_columns_filtered': 0,
            'errors': [],
            'warnings': [],
            'duplicate_ids': 0,
            'processing_time': 0
        }
        self.seen_ids = set()

    def is_target_sheet(self, file_name: str, sheet_name: str) -> bool:
        """Check if this file/sheet combination is in our target list"""
        for target in self.target_sheets:
            if target['file'] == file_name and target['sheet'] == sheet_name:
                return True
        return False

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

    def verify_exact_columns(self, file_path: Path, sheet_name: str) -> bool:
        """Verify that the sheet has EXACTLY the target columns (no more, no less)"""
        try:
            df_sample = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0, engine='openpyxl')

            # Filter to only labeled columns
            labeled_columns = [col for col in df_sample.columns if is_labeled_column(col)]

            # Normalize both target and actual columns for comparison
            normalized_target = [normalize_column_for_pandas(col) for col in self.target_columns]
            normalized_actual = [normalize_column_for_pandas(col) for col in labeled_columns]

            # Check for exact match
            if len(normalized_actual) != len(normalized_target):
                return False

            # Check if all target columns are present and in the right order
            for i, target_col in enumerate(normalized_target):
                if i >= len(normalized_actual) or str(normalized_actual[i]).strip() != str(target_col).strip():
                    return False

            return True

        except Exception as e:
            logger.debug(f"Error verifying columns in sheet '{sheet_name}': {e}")
            return False

    def read_sheet_with_numeric_columns(self, file_path: Path, sheet_name: str) -> pd.DataFrame:
        """Read sheet data handling numeric column names properly"""
        try:
            # Normalize target columns for pandas
            normalized_columns = [normalize_column_for_pandas(col) for col in self.target_columns]

            # First approach: Try reading with the normalized column names
            try:
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    usecols=normalized_columns,
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

                # Filter to only labeled columns first
                labeled_columns = [col for col in df.columns if is_labeled_column(col)]

                # Filter to only the columns we want
                available_cols = []
                for col in labeled_columns:
                    normalized_col = normalize_column_for_pandas(col)
                    if normalized_col in normalized_columns:
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

            # Clean column names to match our target columns exactly
            df.columns = [str(col).strip() for col in df.columns]

            # Remove completely empty rows
            df = df.dropna(how='all')

            logger.debug(f"Read {len(df)} rows from sheet '{sheet_name}' in {file_path.name}")
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

    def process_target_sheet(self, file_path: Path, sheet_name: str) -> List[Dict[str, Any]]:
        """Process a sheet with exact column matching"""
        df = self.read_sheet_with_numeric_columns(file_path, sheet_name)

        if df.empty:
            return []

        # Verify columns still match exactly after reading
        if not self.verify_exact_columns(file_path, sheet_name):
            logger.warning(f"Sheet '{sheet_name}' in {file_path.name} columns don't match exactly after reading")
            return []

        processed_rows = []
        skipped_rows = 0

        # Get file ID
        file_id = self.generate_file_id(file_path)

        for idx, row in df.iterrows():
            try:
                # Build text object - combine all columns as JSON object
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
                    skipped_rows += 1
                    continue

                # Create metadata
                metadata = {
                    'file_name': file_path.name,
                    'file_path': str(file_path.relative_to(self.folder_path)),
                    'sheet_name': sheet_name,
                    'row_index': int(idx),
                    'total_labeled_columns': len(self.target_columns),
                    'non_empty_columns': len(non_empty_columns),
                    'columns_used': non_empty_columns,
                    'text_length': sum(len(str(v)) for v in text_object.values()),
                    'word_count': sum(len(str(v).split()) for v in text_object.values()),
                    'processing_timestamp': datetime.now().isoformat()
                }

                # Add each column as separate metadata field
                for col in df.columns:
                    # Create clean field name for metadata
                    clean_field_name = str(col).lower().replace(' ', '_').replace('-', '_')
                    value = self.clean_text(row[col], col_name=col)
                    metadata[clean_field_name] = value if value else None

                # New record structure with separate id and sheet
                processed_row = {
                    'id': file_id,
                    'sheet': sheet_name,
                    'text': text_object,  # JSON object with preserved formatting
                    'metadata': metadata
                }

                processed_rows.append(processed_row)

            except Exception as e:
                error_msg = f"Error processing row {idx} in sheet '{sheet_name}': {e}"
                logger.debug(error_msg)
                self.processing_stats['errors'].append(error_msg)
                skipped_rows += 1
                continue

        self.processing_stats['rows_skipped'] += skipped_rows

        if processed_rows:
            logger.info(f"    Processed {len(processed_rows)} rows from sheet '{sheet_name}' (skipped {skipped_rows})")

        return processed_rows

    def process_all_files(self, show_progress: bool = True) -> List[Dict[str, Any]]:
        """Process only the exact target files and sheets"""
        start_time = datetime.now()
        all_processed_data = []

        # Group target sheets by file for efficient processing
        files_to_process = defaultdict(list)
        for target in self.target_sheets:
            files_to_process[target['file']].append(target['sheet'])

        if show_progress and HAS_TQDM:
            file_progress = tqdm(files_to_process.items(), desc="Processing files")
        else:
            file_progress = files_to_process.items()

        for file_name, target_sheet_names in file_progress:
            file_path = self.folder_path / file_name

            try:
                if show_progress and HAS_TQDM:
                    file_progress.set_description(f"Processing {file_name}")

                logger.info(f"Processing file: {file_name}")

                # Validate file exists and is readable
                is_valid, error_msg = self.validate_file(file_path)
                if not is_valid:
                    error_msg = f"Skipping {file_name}: {error_msg}"
                    logger.error(error_msg)
                    self.processing_stats['errors'].append(error_msg)
                    self.processing_stats['files_skipped'] += 1
                    continue

                file_processed_rows = 0

                for sheet_name in target_sheet_names:
                    try:
                        logger.info(f"  Processing target sheet: {sheet_name}")

                        # Verify this sheet has the exact columns we expect
                        if self.verify_exact_columns(file_path, sheet_name):
                            # Process the sheet
                            sheet_data = self.process_target_sheet(file_path, sheet_name)
                            all_processed_data.extend(sheet_data)
                            file_processed_rows += len(sheet_data)

                            self.processing_stats['sheets_processed'] += 1
                            self.processing_stats['target_sheets_found'] += 1
                        else:
                            warning_msg = f"Sheet '{sheet_name}' does not have exact column match - skipping"
                            logger.warning(f"    {warning_msg}")
                            self.processing_stats['warnings'].append(f"{file_name} - {warning_msg}")
                            self.processing_stats['sheets_skipped'] += 1

                    except Exception as e:
                        error_msg = f"Error processing sheet '{sheet_name}' in {file_name}: {e}"
                        logger.error(error_msg)
                        self.processing_stats['errors'].append(error_msg)
                        continue

                logger.info(f"File {file_name} complete: {file_processed_rows} total rows processed")
                self.processing_stats['files_processed'] += 1

            except Exception as e:
                error_msg = f"Error processing file {file_name}: {e}"
                logger.error(error_msg)
                self.processing_stats['errors'].append(error_msg)
                continue

        # Calculate processing time
        end_time = datetime.now()
        self.processing_stats['processing_time'] = (end_time - start_time).total_seconds()
        self.processing_stats['rows_processed'] = len(all_processed_data)

        self.processed_data = all_processed_data

        logger.info(f"Processing complete: {len(all_processed_data)} total records from "
                    f"{self.processing_stats['target_sheets_found']} target sheets in "
                    f"{self.processing_stats['processing_time']:.2f} seconds")

        return all_processed_data

    def save_as_readable_jsonl(self) -> str:
        """Save as readable JSONL format with pretty printing"""
        if not self.processed_data:
            logger.warning("No data to save")
            return None

        output_path = self.folder_path / self.output_file

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.processed_data:
                # Write each JSON object with pretty printing (indented format)
                json_line = json.dumps(item, ensure_ascii=False, indent=2)
                f.write(json_line + '\n')

        logger.info(f"Saved {len(self.processed_data)} processed records to {output_path}")
        return str(output_path)

    def save_as_readable_json(self) -> str:
        """Save as readable JSON array format"""
        if not self.processed_data:
            logger.warning("No data to save")
            return None

        # Change extension to .json for array format
        output_file_json = self.output_file.replace('.jsonl', '.json')
        output_path = self.folder_path / output_file_json

        # Save as JSON array with pretty printing
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(self.processed_data)} processed records to {output_path}")
        return str(output_path)

    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary"""
        if not self.processed_data:
            return {'message': 'No data processed yet'}

        # Basic statistics
        files = set()
        sheets = set()
        text_lengths = []

        for item in self.processed_data:
            meta = item['metadata']
            files.add(meta['file_name'])
            sheets.add(f"{meta['file_name']}_{meta['sheet_name']}")
            text_lengths.append(meta['text_length'])

        text_lengths = np.array(text_lengths) if text_lengths else np.array([0])

        summary = {
            'processing_stats': self.processing_stats,
            'data_overview': {
                'total_files': len(files),
                'total_target_sheets': len(sheets),
                'total_records': len(self.processed_data),
                'target_columns': self.target_columns,
                'files_processed': sorted(list(files))
            },
            'text_statistics': {
                'average_text_length': float(np.mean(text_lengths)),
                'median_text_length': float(np.median(text_lengths)),
                'min_text_length': int(np.min(text_lengths)),
                'max_text_length': int(np.max(text_lengths)),
                'total_characters': int(np.sum(text_lengths))
            },
            'quality_metrics': {
                'processing_success_rate': (self.processing_stats['files_processed'] /
                                            max(1, self.processing_stats['files_processed'] + self.processing_stats[
                                                'files_skipped'])) * 100,
                'target_sheets_ratio': 100.0,  # We only process target sheets
                'error_count': len(self.processing_stats['errors']),
                'warning_count': len(self.processing_stats['warnings'])
            },
            'exact_match_data_ready': True,
            'output_format': 'JSONL',
            'ready_for_embeddings': len(self.processed_data) > 0
        }

        return summary


def main():
    """Main function with analysis and selection"""

    print("🚀 Excel Data Converter with Exact Column Matching")
    print("=" * 60)
    print("This tool analyzes Excel files to find sheets with identical column patterns,")
    print("then extracts data from sheets with the exact column pattern you select.")
    print("• Handles numeric column names (including negative numbers)")
    print("• Preserves line breaks and formatting for code samples")
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
        return None, None

    if not os.path.exists(folder_path):
        print(f"❌ Error: Path does not exist: {folder_path}")
        return None, None

    try:
        # Step 1: Analyze column patterns
        print("🔍 Analyzing column patterns in Excel files...")
        analysis_results = count_sheet_column_names(folder_path)

        if not analysis_results:
            print("❌ No analysis results found. Please check your Excel files.")
            return None, None

        # Step 2: Get user selection
        print("\nEnter the rank number to extract data from:")
        while True:
            try:
                rank_choice = int(input("Rank number: ").strip())
                if rank_choice in analysis_results:
                    break
                else:
                    print(f"❌ Invalid rank. Please enter a number between 1 and {len(analysis_results)}")
            except ValueError:
                print("❌ Please enter a valid number")

        selected_pattern = analysis_results[rank_choice]
        target_columns = selected_pattern['columns']
        target_sheets = selected_pattern['occurrences']

        print(f"\n✅ Selected rank {rank_choice} with columns: {target_columns}")
        print(f"📊 Will process {selected_pattern['count']} sheets")

        # Step 3: Get output filename
        print("\nEnter the name for the output .jsonl file (without extension):")
        output_filename = input("Filename: ").strip()
        if not output_filename:
            output_filename = f"rank_{rank_choice}_data"

        # Ensure .jsonl extension
        if not output_filename.endswith('.jsonl'):
            output_filename += '.jsonl'

        print(f"📁 Output will be saved as: {output_filename}")

        # Step 4: Process the data
        converter = ExcelDataConverter(
            folder_path=folder_path,
            target_columns=target_columns,
            target_sheets=target_sheets,
            output_file=output_filename,
            chunk_size=1000,
            max_file_size_mb=100.0
        )

        print(f"\n🚀 Processing {len(target_sheets)} target sheets...")
        processed_data = converter.process_all_files(show_progress=True)

        if processed_data:
            # Save the data
            output_file = converter.save_as_readable_jsonl()

            # Get summary
            summary = converter.get_data_summary()

            # Display results
            print("\n" + "=" * 80)
            print("🎉 DATA EXTRACTION COMPLETE")
            print("=" * 80)

            print(f"✅ Successfully processed {len(processed_data)} records")
            print(f"📁 Output file: {output_file}")
            print(f"📊 Target sheets processed: {summary['processing_stats']['target_sheets_found']}")
            print(f"📄 Files processed: {summary['data_overview']['total_files']}")
            print(f"📝 Columns extracted: {', '.join(target_columns)}")
            print(f"⏱️  Processing time: {summary['processing_stats']['processing_time']:.2f} seconds")
            print(f"📏 Average text length: {summary['text_statistics']['average_text_length']:.1f} characters")

            if summary['processing_stats']['errors']:
                print(f"\n❌ Errors encountered: {len(summary['processing_stats']['errors'])}")
                print("Specific errors:")
                for error in summary['processing_stats']['errors'][:10]:  # Show first 10 errors
                    print(f"   • {error}")
                if len(summary['processing_stats']['errors']) > 10:
                    print(f"   ... and {len(summary['processing_stats']['errors']) - 10} more errors")

            if summary['processing_stats']['warnings']:
                print(f"\n⚠️  Warnings: {len(summary['processing_stats']['warnings'])}")
                print("Specific warnings:")
                for warning in summary['processing_stats']['warnings'][:10]:  # Show first 10 warnings
                    print(f"   • {warning}")
                if len(summary['processing_stats']['warnings']) > 10:
                    print(f"   ... and {len(summary['processing_stats']['warnings']) - 10} more warnings")

            print(f"\n✨ Data Quality:")
            print(f"   • Processing success rate: {summary['quality_metrics']['processing_success_rate']:.1f}%")
            print(f"   • Exact column matches: 100% (by design)")
            print(f"   • Line breaks preserved for code samples")

            print(f"\n📋 Next steps:")
            print(f"   1. Install sentence-transformers: pip install sentence-transformers")
            print(f"   2. Generate embeddings from: {output_file}")
            print(f"   3. Load embeddings into Milvus vector database")

            print(f"\n💡 The JSONL file contains data from sheets with exactly these columns:")
            for col in target_columns:
                print(f"   • {col}")

            return output_file, summary
        else:
            print("❌ No data was processed.")
            print("   • Check if the selected sheets contain valid data")
            print("   • Verify the Excel files are readable")
            print("   • Check the log messages above for specific errors")
            return None, None

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"❌ Error: {e}")
        return None, None


if __name__ == "__main__":
    main()
