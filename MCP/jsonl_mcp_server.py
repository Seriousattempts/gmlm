#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp>=1.2.0,<2",
# ]
# ///
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)
mcp = FastMCP("jsonl-mcp-server")


class JSONLMCPHandler:
    def __init__(self, data_dir: str = "."):
        """Initialize handler with data directory relative to launch folder"""
        self.data_dir = Path(data_dir)
        self.current_jsonl: Optional[str] = None
        self.records: List[Dict[str, Any]] = []
        self.schema: Dict[str, Any] = {}
        self.text_fields: List[str] = []
        self.metadata_fields: List[str] = ["id", "sheet"]

    def load_jsonl_file(self, jsonl_file_path: str) -> Dict[str, Any]:
        """
        Load a JSONL file and its corresponding schema file from data_dir.
        Expects a pair:
          - <name>.jsonl
          - <name>_texted.txt  (JSON schema with {"text": {"text_fields": [...]}})
        """
        jsonl_path = self.data_dir / jsonl_file_path
        if not jsonl_path.exists():
            return {"error": f"JSONL file not found: {jsonl_file_path} in {self.data_dir}"}

        schema_file_path = self.data_dir / f"{jsonl_path.stem}_texted.txt"
        if not schema_file_path.exists():
            return {"error": f"Schema file not found: {schema_file_path}"}

        try:
            # Load schema
            with open(schema_file_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)

            self.text_fields = self.schema.get("text", {}).get("text_fields", [])

            # Load JSONL records
            self.records = self._parse_jsonl_file(str(jsonl_path))
            self.current_jsonl = str(jsonl_path)

            # Auto-discover metadata fields from first record
            if self.records:
                first_record = self.records[0]
                self.metadata_fields = [k for k in first_record.keys() if k != "text"]

            return {
                "message": f"Successfully loaded {jsonl_path.name}",
                "records_count": len(self.records),
                "text_fields": self.text_fields,
                "total_records": len(self.records),
            }
        except Exception as e:
            return {"error": f"Failed to load files: {str(e)}"}

    def _parse_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse JSONL file handling both line-by-line and concatenated formats"""
        records: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Try standard JSONL first (line by line)
        if "\n" in content:
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                    continue
                except json.JSONDecodeError:
                    # Fall through to concatenated parsing
                    break

        # If standard JSONL failed, try concatenated JSON parsing
        if not records:
            records = self._parse_concatenated_json(content)

        return records

    def _parse_concatenated_json(self, content: str) -> List[Dict[str, Any]]:
        """Parse concatenated JSON objects without delimiters"""
        records: List[Dict[str, Any]] = []
        current_object = ""
        brace_depth = 0
        in_string = False
        escape_next = False

        for char in content:
            current_object += char

            if char == '"' and not escape_next:
                in_string = not in_string
            elif char == "\\" and in_string:
                escape_next = not escape_next
                continue

            if not in_string:
                if char == "{":
                    brace_depth += 1
                elif char == "}":
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

            # FIXED: Reset escape flag properly
            if char != "\\":
                escape_next = False

        return records

    def get_available_files(self) -> Dict[str, List[str]]:
        """List available .jsonl and .txt files in data_dir"""
        if not self.data_dir.exists():
            return {"jsonl_files": [], "txt_files": []}

        jsonl_files = [f.name for f in self.data_dir.glob("*.jsonl")]
        txt_files = [f.name for f in self.data_dir.glob("*.txt")]

        return {"jsonl_files": jsonl_files, "txt_files": txt_files}

    def search_records(self, query: str = "", filters: Dict[str, str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search loaded records with optional filters.
        - filters may include "text.<FieldName>" and metadata fields present in records
        """
        if not self.current_jsonl:
            return [{"error": "No JSONL file loaded"}]

        filtered_records = self.records.copy()
        filters = filters or {}

        # Apply filters
        for field, value in filters.items():
            if field.startswith("text."):
                field_name = field[5:]
                if field_name in self.text_fields:
                    val_lower = str(value).lower()
                    filtered_records = [
                        r for r in filtered_records
                        if val_lower in str(r.get("text", {}).get(field_name, "")).lower()
                    ]
            elif field in self.metadata_fields:
                val_lower = str(value).lower()
                filtered_records = [
                    r for r in filtered_records
                    if val_lower in str(r.get(field, "")).lower()
                ]

        # Full-text search across text_fields
        if query:
            ql = query.lower()
            filtered_records = [
                r for r in filtered_records
                if any(ql in str(r.get("text", {}).get(tf, "")).lower() for tf in self.text_fields)
            ]

        return filtered_records[:limit]


# Single handler instance; data_dir configurable via CLI
handler = JSONLMCPHandler()


# ----------------------------
# MCP Tools (Returning structured dict data)
# ----------------------------

@mcp.tool()
def health() -> dict:
    """
    Health check and environment info showing current status and available files.
    """
    available = handler.get_available_files()
    return {
        "status": "healthy",
        "current_file": Path(handler.current_jsonl).name if handler.current_jsonl else None,
        "available_files": available,
        "records_loaded": len(handler.records),
        "data_directory": str(handler.data_dir.resolve()),
    }


@mcp.tool()
def list_files() -> dict:
    """
    List available JSONL and TXT files in the data directory.
    """
    available = handler.get_available_files()
    return {
        "jsonl_files": available["jsonl_files"],
        "txt_files": available["txt_files"],
        "message": f"Found {len(available['jsonl_files'])} JSONL and {len(available['txt_files'])} TXT files",
    }


@mcp.tool()
def load_jsonl(file_path: str) -> dict:
    """
    Load a JSONL file and its paired _texted.txt schema from the data directory.

    Args:
        file_path: Name of the JSONL file in the data directory

    Returns:
        Success response with loading details or error message
    """
    result = handler.load_jsonl_file(file_path)
    if "error" in result:
        return {"success": False, "error": result["error"]}

    return {
        "success": True,
        "data": result,
        "message": f"Loaded {result['records_count']} records. Text fields: {', '.join(result['text_fields'])}",
    }


@mcp.tool()
def get_schema() -> dict:
    """
    Get schema and dataset info for the currently loaded file.
    """
    if not handler.current_jsonl:
        return {"success": False, "error": "No JSONL file loaded. Call load_jsonl first."}

    info = {
        "current_file": Path(handler.current_jsonl).name,
        "text_fields": handler.text_fields,
        "metadata_fields": handler.metadata_fields,
        "total_records": len(handler.records),
    }
    return {"success": True, "data": info, "message": f"Schema for {info['current_file']}"}


@mcp.tool()
def search_data(
        query: str = "",
        text_function: str = "",
        text_returns: str = "",
        sheet: str = "",
        limit: int = 5
) -> dict:
    """
    Search loaded data with optional filters.

    Args:
        query: Full-text search across all text fields
        text_function: Filter by text.Function field
        text_returns: Filter by text.Returns field
        sheet: Filter by metadata 'sheet' field
        limit: Maximum results to return (1-50)

    Returns:
        Search results with metadata or error message
    """
    if not handler.current_jsonl:
        return {"success": False, "error": "No JSONL file loaded. Call load_jsonl first."}

    # Validate and clamp limit
    limit = max(1, min(50, limit))

    # Build filters
    filters = {}
    if text_function:
        filters["text.Function"] = text_function
    if text_returns:
        filters["text.Returns"] = text_returns
    if sheet:
        filters["sheet"] = sheet

    results = handler.search_records(query=query, filters=filters, limit=limit)

    if not results:
        return {
            "success": True,
            "data": {"results": [], "count": 0, "query": query, "filters": filters},
            "message": "No results found."
        }

    # FIXED: Corrected error handling bug - check first element, not the list
    if isinstance(results[0], dict) and "error" in results:
        return {"success": False, "error": results.get("error", "Unknown error")}

    return {
        "success": True,
        "data": {"results": results, "count": len(results), "query": query, "filters": filters},
        "message": f"Found {len(results)} results",
    }


@mcp.tool()
def get_full_record(record_id: str) -> dict:
    """
    Retrieve a complete record by 'id' from the loaded dataset.

    Args:
        record_id: The ID of the record to retrieve

    Returns:
        Complete record data or error message
    """
    if not handler.current_jsonl:
        return {"success": False, "error": "No JSONL file loaded. Call load_jsonl first."}

    record = next((r for r in handler.records if r.get("id") == record_id), None)
    if not record:
        return {"success": False, "error": f"Record not found: {record_id}"}

    return {"success": True, "data": record, "message": f"Retrieved record: {record_id}"}


@mcp.tool()
def search_by_id_pattern(pattern: str, limit: int = 10) -> dict:
    """
    Search for records by ID pattern matching (case-insensitive).

    Args:
        pattern: Pattern to match in record IDs
        limit: Maximum results to return (1-50)

    Returns:
        Matching records or error message
    """
    if not handler.current_jsonl:
        return {"success": False, "error": "No JSONL file loaded. Call load_jsonl first."}

    # Validate and clamp limit
    limit = max(1, min(50, limit))
    pattern_lower = pattern.lower()

    matching_records = [
                           record for record in handler.records
                           if pattern_lower in str(record.get('id', '')).lower()
                       ][:limit]

    if not matching_records:
        return {
            "success": True,
            "data": {"results": [], "count": 0, "pattern": pattern},
            "message": f"No records found with ID pattern: '{pattern}'"
        }

    return {
        "success": True,
        "data": {"results": matching_records, "count": len(matching_records), "pattern": pattern},
        "message": f"Found {len(matching_records)} records matching ID pattern: '{pattern}'"
    }


@mcp.tool()
def get_record_summary(limit: int = 10) -> dict:
    """
    Get a summary of loaded records showing basic info for quick browsing.

    Args:
        limit: Maximum number of record summaries to return (1-100)

    Returns:
        Summary information for records or error message
    """
    if not handler.current_jsonl:
        return {"success": False, "error": "No JSONL file loaded. Call load_jsonl first."}

    # Validate and clamp limit
    limit = max(1, min(100, limit))

    summaries = []
    for record in handler.records[:limit]:
        summary = {
            "id": record.get("id", "N/A"),
        }

        # Add metadata fields
        for field in handler.metadata_fields:
            if field != "id" and field in record:
                summary[field] = record[field]

        # Add preview of first text field
        if "text" in record and handler.text_fields:
            first_field = handler.text_fields[0]
            if first_field in record["text"]:
                preview = str(record["text"][first_field])[:100]
                if len(str(record["text"][first_field])) > 100:
                    preview += "..."
                summary[f"{first_field}_preview"] = preview

        summaries.append(summary)

    return {
        "success": True,
        "data": {
            "summaries": summaries,
            "count": len(summaries),
            "total_records": len(handler.records)
        },
        "message": f"Retrieved {len(summaries)} record summaries (of {len(handler.records)} total)"
    }


def main():
    parser = argparse.ArgumentParser(description="MCP-native JSONL server (FastMCP)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Subfolder with .jsonl and _texted.txt files (default: ./data)")
    parser.add_argument("--transport", type=str, default="stdio", choices=["stdio", "sse"],
                        help="MCP transport protocol (stdio|sse). Default: stdio")
    parser.add_argument("--mcp-host", type=str, default="127.0.0.1",
                        help="Host for SSE transport (default: 127.0.0.1)")
    parser.add_argument("--mcp-port", type=int, default=8081,
                        help="Port for SSE transport (default: 8081)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    mcp.settings.log_level = args.log_level

    # FIXED: Configure data directory with absolute path resolution
    handler.data_dir = Path(args.data_dir).resolve()
    logger.info(f"Data directory: {handler.data_dir}")

    # Ensure data directory exists
    handler.data_dir.mkdir(parents=True, exist_ok=True)

    # Run server
    if args.transport == "sse":
        mcp.settings.host = args.mcp_host
        mcp.settings.port = args.mcp_port
        logger.info(f"Starting MCP server (SSE) at http://{mcp.settings.host}:{mcp.settings.port}/sse")
        mcp.run(transport="sse")
    else:
        logger.info("Starting MCP server (stdio)")
        mcp.run()


if __name__ == "__main__":
    main()
