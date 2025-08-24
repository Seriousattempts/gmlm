from fastapi import FastAPI, HTTPException, Query, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path as PathLib
import json
import re
from datetime import datetime
import uvicorn


# Pydantic Models
class SearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="Full-text search query")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Dynamic filters")
    fields: Optional[List[str]] = Field(None, description="Specific text fields to return")
    limit: int = Field(20, ge=1, le=100, description="Number of results to return")
    offset: int = Field(0, ge=0, description="Number of results to skip")


class APIResponse(BaseModel):
    data: Any
    meta: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None


# Main JSONL API Handler
class JSONLAPIHandler:
    def __init__(self):
        self.current_jsonl: Optional[str] = None
        self.records: List[Dict[str, Any]] = []
        self.schema: Dict[str, Any] = {}
        self.text_fields: List[str] = []
        self.metadata_fields: List[str] = ["id", "sheet"]

    def load_jsonl_file(self, jsonl_file_path: str) -> bool:
        """Load JSONL file and its corresponding schema file"""
        jsonl_path = PathLib(jsonl_file_path)

        if not jsonl_path.exists():
            raise HTTPException(status_code=404, detail=f"JSONL file not found: {jsonl_file_path}")

        # Load corresponding schema file
        schema_file_path = jsonl_path.with_name(f"{jsonl_path.stem}_texted.txt")
        if not schema_file_path.exists():
            raise HTTPException(status_code=404, detail=f"Schema file not found: {schema_file_path}")

        try:
            # Load schema
            with open(schema_file_path, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
                self.text_fields = self.schema.get("text", {}).get("text_fields", [])

            # Load JSONL records
            self.records = self._parse_jsonl_file(str(jsonl_path))
            self.current_jsonl = str(jsonl_path)

            # Auto-discover metadata fields from first record
            if self.records:
                first_record = self.records[0]
                self.metadata_fields = [key for key in first_record.keys() if key != "text"]

            return True

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load files: {str(e)}")

    def _parse_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse JSONL file handling both line-by-line and concatenated formats"""
        records = []

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Try standard JSONL first (line by line)
        if '\n' in content:
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                    continue
                except json.JSONDecodeError:
                    break

        # If standard JSONL failed, try concatenated JSON parsing
        if not records:
            records = self._parse_concatenated_json(content)

        return records

    def _parse_concatenated_json(self, content: str) -> List[Dict[str, Any]]:
        """Parse concatenated JSON objects"""
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

    def get_schema_info(self) -> Dict[str, Any]:
        """Get schema information for the loaded JSONL"""
        if not self.current_jsonl:
            raise HTTPException(status_code=400, detail="No JSONL file loaded")

        return {
            "current_file": PathLib(self.current_jsonl).name,
            "text_fields": self.text_fields,
            "metadata_fields": self.metadata_fields,
            "total_records": len(self.records),
            "sample_record": self.records[0] if self.records else None
        }

    def filter_records(self, filters: Dict[str, Any], query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter records based on dynamic filters and search query"""
        filtered_records = self.records.copy()

        # Apply text field filters
        text_filters = filters.get("text", {})
        for field, value in text_filters.items():
            if field in self.text_fields:
                filtered_records = [
                    record for record in filtered_records
                    if record.get("text", {}).get(field, "").lower().find(str(value).lower()) != -1
                ]

        # Apply metadata filters
        metadata_filters = {k: v for k, v in filters.items() if k != "text"}
        for field, value in metadata_filters.items():
            if field in self.metadata_fields:
                filtered_records = [
                    record for record in filtered_records
                    if str(record.get(field, "")).lower().find(str(value).lower()) != -1
                ]

        # Apply full-text search across all text fields
        if query:
            query_lower = query.lower()
            filtered_records = [
                record for record in filtered_records
                if any(
                    query_lower in str(record.get("text", {}).get(field, "")).lower()
                    for field in self.text_fields
                )
            ]

        return filtered_records

    def format_record(self, record: Dict[str, Any], requested_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Format record with optional field selection"""
        formatted = {
            "id": record.get("id"),
            "sheet": record.get("sheet"),
            "text": {},
            "metadata": {}
        }

        # Add metadata fields
        for field in self.metadata_fields:
            if field in record and field not in ["id", "sheet"]:
                formatted["metadata"][field] = record[field]

        # Include requested text fields or all if none specified
        text_fields_to_include = requested_fields or self.text_fields
        for field in text_fields_to_include:
            if field in self.text_fields and field in record.get("text", {}):
                formatted["text"][field] = record["text"][field]

        return formatted


# FastAPI App
app = FastAPI(
    title="Dynamic JSONL API",
    version="1.0.0",
    description="Schema-agnostic API for JSONL files with automatic schema discovery"
)

# Global handler
handler = JSONLAPIHandler()


@app.post("/v1/load/{file_name}")
async def load_jsonl_file(file_name: str = Path(..., description="Name of JSONL file to load")):
    """Load a specific JSONL file and its schema"""
    try:
        success = handler.load_jsonl_file(file_name)
        return APIResponse(
            data={
                "message": f"Successfully loaded {file_name}",
                "records_count": len(handler.records),
                "text_fields": handler.text_fields
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/schema")
async def get_schema():
    """Get schema information for the currently loaded JSONL file"""
    try:
        schema_info = handler.get_schema_info()
        return APIResponse(data=schema_info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/records")
async def get_records(
        request: Request,
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Records per page"),
        fields: Optional[str] = Query(None, description="Comma-separated text fields to return"),
        q: Optional[str] = Query(None, description="Full-text search query")
):
    """Get records with dynamic filtering and pagination"""
    if not handler.current_jsonl:
        raise HTTPException(status_code=400, detail="No JSONL file loaded. Use POST /v1/load/{file_name} first")

    # Parse dynamic query parameters
    query_params = dict(request.query_params)

    # Separate text filters (text.field_name) from metadata filters
    filters = {"text": {}}
    for key, value in query_params.items():
        if key.startswith("text."):
            field_name = key[5:]  # Remove "text." prefix
            if field_name in handler.text_fields:
                filters["text"][field_name] = value
        elif key in handler.metadata_fields:
            filters[key] = value

    # Filter records
    filtered_records = handler.filter_records(filters, q)

    # Pagination
    start = (page - 1) * page_size
    end = start + page_size
    page_records = filtered_records[start:end]

    # Field selection
    requested_fields = fields.split(",") if fields else None
    if requested_fields:
        # Validate requested fields
        invalid_fields = [f for f in requested_fields if f not in handler.text_fields]
        if invalid_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid text fields: {invalid_fields}. Available fields: {handler.text_fields}"
            )

    # Format records
    formatted_records = [
        handler.format_record(record, requested_fields)
        for record in page_records
    ]

    return APIResponse(
        data=formatted_records,
        meta={
            "page": page,
            "page_size": page_size,
            "total": len(filtered_records),
            "total_pages": (len(filtered_records) + page_size - 1) // page_size,
            "current_file": PathLib(handler.current_jsonl).name,
            "filters_applied": filters,
            "search_query": q
        }
    )


@app.get("/v1/records/{record_id}")
async def get_record_by_id(record_id: str = Path(..., description="Record ID")):
    """Get a single record by ID"""
    if not handler.current_jsonl:
        raise HTTPException(status_code=400, detail="No JSONL file loaded")

    record = next((r for r in handler.records if r.get("id") == record_id), None)
    if not record:
        raise HTTPException(status_code=404, detail=f"Record not found: {record_id}")

    return APIResponse(data=handler.format_record(record))


@app.post("/v1/search")
async def search_records(search_request: SearchRequest):
    """Advanced search with complex filtering"""
    if not handler.current_jsonl:
        raise HTTPException(status_code=400, detail="No JSONL file loaded")

    # Validate requested fields
    if search_request.fields:
        invalid_fields = [f for f in search_request.fields if f not in handler.text_fields]
        if invalid_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid text fields: {invalid_fields}. Available fields: {handler.text_fields}"
            )

    # Separate text and metadata filters
    filters = {"text": {}}
    for key, value in search_request.filters.items():
        if key.startswith("text."):
            field_name = key[5:]
            if field_name in handler.text_fields:
                filters["text"][field_name] = value
        elif key in handler.metadata_fields:
            filters[key] = value

    # Filter records
    filtered_records = handler.filter_records(filters, search_request.query)

    # Apply limit and offset
    page_records = filtered_records[search_request.offset:search_request.offset + search_request.limit]

    # Format records
    formatted_records = [
        handler.format_record(record, search_request.fields)
        for record in page_records
    ]

    return APIResponse(
        data=formatted_records,
        meta={
            "total": len(filtered_records),
            "offset": search_request.offset,
            "limit": search_request.limit,
            "current_file": PathLib(handler.current_jsonl).name
        }
    )


@app.get("/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "loaded_file": PathLib(handler.current_jsonl).name if handler.current_jsonl else None,
        "records_count": len(handler.records)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
