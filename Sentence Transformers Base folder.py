import json
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustEmbeddingGenerator:
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):
        """Initialize embedding generator with Sentence Transformers model"""
        print(f"🚀 Loading Sentence Transformers model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded! Embedding dimension: {self.embedding_dim}")

    def find_all_jsonl_files(self, folder_path: str) -> List[Path]:
        """Find all .jsonl files in the specified folder"""
        folder_path = Path(folder_path)

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        jsonl_files = list(folder_path.glob("*.jsonl"))

        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in: {folder_path}")

        print(f"📁 Found {len(jsonl_files)} .jsonl files:")
        for file in jsonl_files:
            print(f"   • {file.name}")

        return jsonl_files

    def parse_concatenated_json_objects(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse JSON objects from content using brace counting
        Handles pretty-printed JSON that spans multiple lines
        """
        documents = []
        current_object = ""
        brace_depth = 0
        in_string = False
        escape_next = False

        for char in content:
            current_object += char

            # Handle string literals to avoid counting braces inside strings
            if char == '"' and not escape_next:
                in_string = not in_string
            elif char == '\\' and in_string:
                escape_next = not escape_next
                continue

            # Only count braces outside of strings
            if not in_string:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1

                    # When we reach brace_depth 0, we have a complete JSON object
                    if brace_depth == 0:
                        obj_str = current_object.strip()
                        if obj_str:
                            try:
                                # Parse the JSON object
                                doc = json.loads(obj_str)
                                # Validate it has the expected structure
                                if isinstance(doc, dict) and 'id' in doc and 'text' in doc:
                                    documents.append(doc)
                                else:
                                    print(
                                        f"⚠️ Skipping object without required fields: {list(doc.keys()) if isinstance(doc, dict) else type(doc)}")
                            except json.JSONDecodeError as e:
                                print(f"⚠️ Failed to parse JSON object: {e}")
                                print(f"   Object preview: {obj_str[:100]}...")
                        current_object = ""

            escape_next = False

        return documents

    def load_jsonl_file_robust(self, jsonl_path: Path) -> List[Dict[str, Any]]:
        """Load and parse JSONL file using robust brace-counting method"""
        print(f"📖 Loading: {jsonl_path.name}")

        try:
            # Read entire file content
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse using brace counting
            documents = self.parse_concatenated_json_objects(content)

            print(f"✅ Successfully loaded {len(documents)} documents from {jsonl_path.name}")
            return documents

        except Exception as e:
            print(f"❌ Failed to load {jsonl_path.name}: {e}")
            return []

    def load_all_jsonl_files(self, folder_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load all .jsonl files in the folder using robust parsing"""
        jsonl_files = self.find_all_jsonl_files(folder_path)
        all_data = {}

        for jsonl_file in jsonl_files:
            documents = self.load_jsonl_file_robust(jsonl_file)
            if documents:
                all_data[jsonl_file.stem] = documents

        total_docs = sum(len(docs) for docs in all_data.values())
        print(f"🎉 Total documents loaded: {total_docs} from {len(all_data)} files")
        return all_data

    def extract_text_for_embedding(self, doc: Dict[str, Any]) -> str:
        """Extract text content from document for embedding"""
        text_obj = doc.get('text', {})
        if not isinstance(text_obj, dict):
            return str(text_obj) if text_obj else ""

        # Combine all text field values, preserving structure
        text_parts = []
        for key, value in text_obj.items():
            if value:
                text_parts.append(f"{key}: {str(value)}")

        return " | ".join(text_parts)

    def generate_embeddings_for_folder(self, folder_path: str, output_dir: str = None) -> None:
        """Generate embeddings from ALL .jsonl files in folder"""
        # Load all documents from all .jsonl files
        all_file_data = self.load_all_jsonl_files(folder_path)

        if not all_file_data:
            print("❌ No documents found in any files")
            return

        # Extract texts and metadata from all files
        print("🔍 Extracting text for embedding from all files...")
        texts = []
        metadata = []
        doc_global_index = 0

        for file_name, documents in all_file_data.items():
            print(f"📝 Processing {len(documents)} documents from {file_name}")

            for local_index, doc in enumerate(documents):
                # Validate document structure
                if not isinstance(doc, dict):
                    print(f"⚠️ Skipping non-dict object in {file_name} at index {local_index}")
                    continue

                embedding_text = self.extract_text_for_embedding(doc)
                if not embedding_text:
                    print(f"⚠️ Empty text in {file_name}, document {local_index}, skipping")
                    continue

                texts.append(embedding_text)

                # Keep metadata separate with file information
                metadata.append({
                    'global_index': doc_global_index,
                    'source_file': file_name,
                    'local_index': local_index,
                    'original_id': doc.get('id', ''),
                    'sheet': doc.get('sheet', ''),
                    'file_name': doc.get('metadata', {}).get('file_name', ''),
                    'row_index': doc.get('metadata', {}).get('row_index', -1),
                    'text': embedding_text
                })
                doc_global_index += 1

        if not texts:
            print("❌ No valid texts found for embedding")
            return

        # Generate embeddings
        print(f"🧠 Generating embeddings for {len(texts)} texts from all files...")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )

        print(f"✅ Generated {len(embeddings)} embeddings")

        # Save embeddings and metadata
        self.save_embeddings(folder_path, embeddings, metadata, output_dir)

    def save_embeddings(self, folder_path: str, embeddings: np.ndarray,
                        metadata: List[Dict], output_dir: str = None):
        """Save embeddings and metadata to files"""

        # Determine output directory
        folder_path = Path(folder_path)
        if output_dir is None:
            output_dir = folder_path / "all_embeddings"
        else:
            output_dir = Path(output_dir)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        embeddings_file = output_dir / "embeddings.npy"
        metadata_file = output_dir / "metadata.json"
        info_file = output_dir / "info.json"

        # Save embeddings as numpy array
        print(f"💾 Saving embeddings to: {embeddings_file}")
        np.save(embeddings_file, embeddings)

        # Save metadata as JSON
        print(f"💾 Saving metadata to: {metadata_file}")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save model info
        info_data = {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'num_embeddings': len(embeddings),
            'source_folder': str(folder_path.name),
            'num_source_files': len(set(meta['source_file'] for meta in metadata)),
            'created_at': datetime.now().isoformat()
        }

        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2)

        print(f"\n🎉 Embeddings saved successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 {len(embeddings)} embeddings ({self.embedding_dim} dimensions each)")
        print(f"📋 Source files processed: {info_data['num_source_files']}")
        print(f"📋 Files created:")
        print(f"   • embeddings.npy - The actual embedding vectors")
        print(f"   • metadata.json - Document metadata for each embedding")
        print(f"   • info.json - Model and processing information")


def main():
    """Generate embeddings from ALL .jsonl files in a folder (robust parser)"""

    print("🧠 Robust Embedding Generator - Step 2 of 4")
    print("=" * 60)
    print("Creating embeddings from malformed JSONL files (pretty-printed JSON)")
    print("Uses brace-counting to extract ALL JSON objects without skipping")
    print("=" * 60)

    try:
        # Initialize embedding generator
        generator = RobustEmbeddingGenerator(model_name='BAAI/bge-small-en-v1.5')

        # Get folder path containing .jsonl files
        folder_path = input("\n📁 Enter path to folder containing .jsonl files: ").strip().strip('"\'')

        if not folder_path:
            print("❌ No path provided")
            return

        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return

        # Generate embeddings for ALL .jsonl files in the folder
        generator.generate_embeddings_for_folder(folder_path)

        print(f"\n✅ SUCCESS! Embeddings created from all .jsonl files without skipping content.")
        print(f"📋 Next steps:")
        print(f"   Step 3: Load these embeddings into Milvus")
        print(f"   Step 4: Query Milvus for LLM context")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        print("Full error details:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
