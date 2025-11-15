#!/usr/bin/env python3
"""Huginn Database Management Script.

Populates ChromaDB vector store with chunked scientific papers from Documents folder.
"""

import                  argparse
import                  base64
import                  json
import                  logging
import                  warnings
from pathlib import     Path
from typing import      Iterator
from typing import      Optional
import                  sys

from docling_core.types.doc.document import                         DoclingDocument
from docling_core.types.doc.labels import                           DocItemLabel
from docling_core.transforms.chunker.hierarchical_chunker import    DocMeta
from docling_core.transforms.chunker.hybrid_chunker import          HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import   HuggingFaceTokenizer

from transformers import            AutoTokenizer
from sentence_transformers import   SentenceTransformer

import                  chromadb
from chromadb.config    import Settings

warnings.filterwarnings('ignore', message = 'Migrating deprecated')

def decode_doi_from_filename(filename: str) -> str:
    """Decode DOI from base64-encoded filename.

    Args:
        filename: Base64-encoded DOI (e.g., 'MTAuMjE3NjkvQmlvUHJvdG9jLjE4MTg')

    Returns:
        Decoded DOI (e.g., '10.21769/BioProtoc.1818')
    """

    base_name = Path(filename).stem
    try:
        padding = len(base_name) % 4
        if padding: base_name += '=' * (4 - padding)
        decoded_bytes = base64.b64decode(base_name)
        doi = decoded_bytes.decode('utf-8')
        return doi
    except Exception as e:
        logging.warning(f'Failed to decode DOI from filename {filename}: {e}')
        return base_name

def extract_document_title(doc: DoclingDocument) -> Optional[str]:
    """Extract document title from DoclingDocument.

    Looks for the first section_header in the document.

    Args:
        doc: DoclingDocument instance

    Returns:
        Document title if found, None otherwise
    """
    for text_item in doc.texts:
        if text_item.label == DocItemLabel.SECTION_HEADER: return text_item.text

    # Fallback: try first text item
    if doc.texts and len(doc.texts) > 0: return doc.texts[0].text[:100]  # Truncate if needed

    return None

def extract_authors(doc: DoclingDocument) -> Optional[str]:
    """Attempt to extract author names from document.

    Looks for text patterns near the beginning that might contain authors.

    Args:
        doc: DoclingDocument instance

    Returns:
        Authors string if found, None otherwise
    """
    # Look in first few text items after title
    for text_item in doc.texts[:5]:
        text = text_item.text.strip()

        # Skip if it's a section header or very long
        if text_item.label == DocItemLabel.SECTION_HEADER: continue
        if len(text) > 200: continue

        # Look for patterns that suggest authors
        # Common patterns: names with 'and', asterisks for corresponding author
        if ' and ' in text.lower() or '*' in text:
            # Check if it looks like author names (contains uppercase letters)
            if any(c.isupper() for c in text):
                return text

    return None

def get_page_number(chunk) -> Optional[int]:
    """Extract page number from chunk metadata.

    Args:
        chunk: DocChunk object

    Returns:
        Page number if available, None otherwise
    """
    try:
        if chunk.meta.doc_items and len(chunk.meta.doc_items) > 0:
            first_item = chunk.meta.doc_items[0]
            if first_item.prov and len(first_item.prov) > 0:
                return first_item.prov[0].page_no
    except (AttributeError, IndexError):
        pass

    return None


def load_manifest(manifest_path: Path) -> set[str]:
    """Load existing manifest file and return set of processed filenames.

    Args:
        manifest_path: Path to manifest.tsv file

    Returns:
        Set of base64-encoded filenames that have been processed
    """
    processed = set()
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding = 'utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # First field is the base64 filename
                        fields = line.split('\t')
                        if fields:
                            processed.add(fields[0])
        except Exception as e:
            logger.warning(f'Failed to load manifest: {e}')
    return processed


def load_manifest_full(manifest_path: Path) -> dict[str, tuple[str, str]]:
    """Load manifest file with all fields.

    Args:
        manifest_path: Path to manifest.tsv file

    Returns:
        Dictionary mapping filename -> (doi, title)
    """
    manifest = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding = 'utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        fields = line.split('\t')
                        if len(fields) >= 3:
                            filename, doi, title = fields[0], fields[1], fields[2]
                            manifest[filename] = (doi, title)
        except Exception as e:
            logger.warning(f'Failed to load manifest: {e}')
    return manifest


def write_manifest(manifest_path: Path, manifest: dict[str, tuple[str, str]]):
    """Write complete manifest file.

    Args:
        manifest_path: Path to manifest.tsv file
        manifest: Dictionary mapping filename -> (doi, title)
    """
    try:
        with open(manifest_path, 'w', encoding = 'utf-8') as f:
            for filename, (doi, title) in manifest.items():
                f.write(f'{filename}\t{doi}\t{title}\n')
        logger.info(f'Manifest written with {len(manifest)} entries')
    except Exception as e:
        logger.error(f'Failed to write manifest: {e}')


def append_to_manifest(manifest_path: Path, filename: str, doi: str, title: str):
    """Append entry to manifest file.

    Args:
        manifest_path: Path to manifest.tsv file
        filename: Base64-encoded filename (without .json extension)
        doi: Document DOI
        title: Document title
    """
    try:
        # Sanitize fields for TSV (replace tabs and newlines)
        clean_doi = doi.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        clean_title = title.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')

        with open(manifest_path, 'a', encoding = 'utf-8') as f:
            f.write(f'{filename}\t{clean_doi}\t{clean_title}\n')

        logger.debug(f'Added to manifest: {filename}')
    except Exception as e:
        logger.error(f'Failed to write to manifest: {e}')

# Default configuration
DEFAULT_DOCUMENTS_DIR = Path('Documents')
CHROMA_DB_DIR = Path('Documents_Chroma_DB')
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
MAX_TOKENS = 512
DEFAULT_COLLECTION_NAME = 'scientific_papers'

# Global variables (set by argument parser)
DOCUMENTS_DIR = DEFAULT_DOCUMENTS_DIR
COLLECTION_NAME = DEFAULT_COLLECTION_NAME

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_chunker() -> tuple[HybridChunker, HuggingFaceTokenizer]:
    """Initialize the HybridChunker with appropriate tokenizer.

    Returns:
        Tuple of (chunker, tokenizer)
    """
    logger.info(f'Loading tokenizer: {EMBEDDING_MODEL}')
    tokenizer = HuggingFaceTokenizer(
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
        max_tokens = MAX_TOKENS
    )

    chunker = HybridChunker(tokenizer = tokenizer)
    logger.info('HybridChunker initialized')

    return chunker, tokenizer


def setup_embedding_model() -> SentenceTransformer:
    """Initialize the sentence transformer model for embeddings.

    Returns:
        SentenceTransformer model
    """
    logger.info(f'Loading embedding model: {EMBEDDING_MODEL}')
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def setup_chromadb() -> chromadb.Collection:
    """Initialize ChromaDB with persistent storage.

    Returns:
        ChromaDB collection
    """
    logger.info(f'Setting up ChromaDB at {CHROMA_DB_DIR}')
    CHROMA_DB_DIR.mkdir(exist_ok = True)

    client = chromadb.PersistentClient(
        path = str(CHROMA_DB_DIR),
        settings = Settings(anonymized_telemetry = False)
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name = COLLECTION_NAME,
        metadata = {'description': 'Scientific papers processed with Docling'}
    )

    logger.info(f'Collection "{COLLECTION_NAME}" ready (current count: {collection.count()})')
    return collection


def process_document(
    json_path: Path,
    chunker: HybridChunker,
    _tokenizer: HuggingFaceTokenizer
) -> Iterator[tuple[str, dict, str]]:
    """Process a single document and yield chunks with metadata.

    Args:
        json_path: Path to Docling JSON file
        chunker: HybridChunker instance
        tokenizer: Tokenizer instance

    Yields:
        Tuples of (chunk_text, metadata_dict, chunk_id)
    """
    logger.info(f'Processing {json_path.name}')

    try:
        # Load DoclingDocument
        with open(json_path, 'r') as f:
            doc_dict = json.load(f)

        doc = DoclingDocument.model_validate(doc_dict)

        # Extract document-level metadata
        doi = decode_doi_from_filename(json_path.name)
        title = extract_document_title(doc)
        authors = extract_authors(doc)

        logger.info(f'  DOI: {doi}')
        logger.info(f'  Title: {title}')
        logger.info(f'  Authors: {authors}')

        # Generate chunks
        chunks = list(chunker.chunk(doc))
        logger.info(f'  Generated {len(chunks)} chunks')

        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            # Get contextualized text for embedding
            chunk_text = chunker.contextualize(chunk)

            # Extract chunk metadata
            page_no = get_page_number(chunk)
            # Type narrowing: HybridChunker returns chunks with DocMeta, not BaseMeta
            chunk_meta = chunk.meta
            assert isinstance(chunk_meta, DocMeta), f'Expected DocMeta, got {type(chunk_meta)}'
            headings = chunk_meta.headings if chunk_meta.headings is not None else []

            # Build metadata dictionary
            metadata = {
                'doi': doi,
                'title': title if title else 'Unknown',
                'authors': authors if authors else 'Unknown',
                'chunk_index': chunk_idx,
                'total_chunks': len(chunks),
                'page_number': page_no if page_no else 0,
                'section_headings': ', '.join(headings) if headings else 'None',
                'filename': json_path.name,
                'original_text': chunk.text[:500],  # Store snippet of original text
            }

            # Create unique chunk ID
            chunk_id = f'{doi}__chunk_{chunk_idx}'

            yield chunk_text, metadata, chunk_id

    except Exception as e:
        logger.error(f'Failed to process {json_path.name}: {e}', exc_info = True)


def populate_database(
    collection: chromadb.Collection,
    chunker: HybridChunker,
    tokenizer: HuggingFaceTokenizer,
    embedding_model: SentenceTransformer
):
    """Populate ChromaDB with all documents from Documents folder.

    Args:
        collection: ChromaDB collection
        chunker: HybridChunker instance
        tokenizer: Tokenizer instance
        embedding_model: SentenceTransformer model
    """
    # Setup manifest
    manifest_path = DOCUMENTS_DIR / 'manifest.tsv'
    processed_files = load_manifest(manifest_path)
    logger.info(f'Manifest loaded: {len(processed_files)} files already processed')

    # Find all JSON files
    json_files = list(DOCUMENTS_DIR.glob('*.json'))
    logger.info(f'Found {len(json_files)} JSON files to process')

    if len(json_files) == 0:
        logger.warning(f'No JSON files found in {DOCUMENTS_DIR}')
        return

    total_chunks = 0
    processed_docs = 0
    failed_docs = 0
    skipped_docs = 0

    for json_path in json_files:
        # Check if already processed
        base_filename = json_path.stem
        if base_filename in processed_files:
            logger.info(f'Skipping {json_path.name} (already in manifest)')
            skipped_docs += 1
            continue

        try:
            # Collect chunks from this document
            batch_texts = []
            batch_metadatas = []
            batch_ids = []
            doc_doi = None
            doc_title = None

            for chunk_text, metadata, chunk_id in process_document(json_path, chunker, tokenizer):
                batch_texts.append(chunk_text)
                batch_metadatas.append(metadata)
                batch_ids.append(chunk_id)
                # Capture DOI and title from first chunk
                if doc_doi is None:
                    doc_doi = metadata['doi']
                    doc_title = metadata['title']

            if batch_texts:
                # Generate embeddings
                logger.info(f'  Generating embeddings for {len(batch_texts)} chunks')
                embeddings = embedding_model.encode(
                    batch_texts,
                    show_progress_bar = False,
                    convert_to_numpy = True
                )

                # Add to ChromaDB
                collection.add(
                    embeddings = embeddings.tolist(),
                    documents = batch_texts,
                    metadatas = batch_metadatas,
                    ids = batch_ids
                )

                total_chunks += len(batch_texts)
                processed_docs += 1
                logger.info(f'  ✓ Added {len(batch_texts)} chunks to database')

                # Add to manifest
                append_to_manifest(
                    manifest_path,
                    base_filename,
                    doc_doi if doc_doi else 'Unknown',
                    doc_title if doc_title else 'Unknown'
                )

        except Exception as e:
            failed_docs += 1
            logger.error(f'Failed to process {json_path.name}: {e}')
            continue

    # Summary
    logger.info('='*60)
    logger.info('Database population complete!')
    logger.info(f'  Successfully processed: {processed_docs} documents')
    logger.info(f'  Skipped (already processed): {skipped_docs} documents')
    logger.info(f'  Failed: {failed_docs} documents')
    logger.info(f'  Total chunks added: {total_chunks}')
    logger.info(f'  Final collection count: {collection.count()}')
    logger.info(f'  Manifest location: {manifest_path}')
    logger.info('='*60)


def cleanup_database(collection: chromadb.Collection):
    """Remove database entries for files not in documents directory and update manifest.

    Args:
        collection: ChromaDB collection
    """
    manifest_path = DOCUMENTS_DIR / 'manifest.tsv'
    logger.info('Starting cleanup mode')
    logger.info('='*60)

    # Load current manifest
    manifest = load_manifest_full(manifest_path)
    logger.info(f'Loaded manifest with {len(manifest)} entries')

    # Get current JSON files in directory
    current_files = {f.stem for f in DOCUMENTS_DIR.glob('*.json')}
    logger.info(f'Found {len(current_files)} JSON files in directory')

    # Find orphaned entries (in manifest but not in directory)
    orphaned = set(manifest.keys()) - current_files
    logger.info(f'Found {len(orphaned)} orphaned entries to remove')

    if len(orphaned) == 0:
        logger.info('No cleanup needed - all manifest entries have corresponding files')
        logger.info('='*60)
        return

    # Remove from ChromaDB
    removed_chunks = 0
    for filename in orphaned:
        doi, _title = manifest[filename]
        logger.info(f'Removing: {filename} (DOI: {doi})')

        try:
            # Find all chunk IDs for this document
            # Chunks are named like: {doi}__chunk_{idx}
            # We need to query for all chunks with this DOI prefix
            results = collection.get(
                where = {'doi': doi},
                include = []
            )

            if results and results['ids']:
                chunk_ids = results['ids']
                logger.info(f'  Found {len(chunk_ids)} chunks to remove')

                # Delete the chunks
                collection.delete(ids = chunk_ids)
                removed_chunks += len(chunk_ids)
                logger.info(f'  ✓ Removed {len(chunk_ids)} chunks')
            else:
                logger.warning(f'  No chunks found for DOI: {doi}')

        except Exception as e:
            logger.error(f'  Failed to remove chunks for {filename}: {e}')

    # Update manifest to remove orphaned entries
    for filename in orphaned:
        del manifest[filename]

    write_manifest(manifest_path, manifest)

    # Summary
    logger.info('='*60)
    logger.info('Cleanup complete!')
    logger.info(f'  Removed entries from manifest: {len(orphaned)}')
    logger.info(f'  Removed chunks from database: {removed_chunks}')
    logger.info(f'  Remaining manifest entries: {len(manifest)}')
    logger.info(f'  Final collection count: {collection.count()}')
    logger.info('='*60)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description = 'Huginn Database Management - Populate ChromaDB with scientific papers'
    )
    parser.add_argument(
        '-d', '--documents',
        type = Path,
        default = DEFAULT_DOCUMENTS_DIR,
        help = f'Path to documents directory (default: {DEFAULT_DOCUMENTS_DIR})'
    )
    parser.add_argument(
        '-c', '--collection',
        type = str,
        default = DEFAULT_COLLECTION_NAME,
        help = f'ChromaDB collection name (default: {DEFAULT_COLLECTION_NAME})'
    )
    parser.add_argument(
        '--cleanup',
        action = 'store_true',
        help = 'Remove database entries for files not in documents directory and update manifest'
    )
    return parser.parse_args()


def main():
    """Main entry point for Huginn_DB."""
    global DOCUMENTS_DIR, COLLECTION_NAME

    # Parse command line arguments
    args = parse_arguments()
    DOCUMENTS_DIR = args.documents
    COLLECTION_NAME = args.collection

    logger.info('Starting Huginn Database Management')
    logger.info('='*60)
    logger.info(f'Documents directory: {DOCUMENTS_DIR}')
    logger.info(f'Collection name: {COLLECTION_NAME}')

    # Check if Documents directory exists
    if not DOCUMENTS_DIR.exists():
        logger.error(f'Documents directory not found: {DOCUMENTS_DIR}')
        sys.exit(1)

    # Setup ChromaDB (always needed)
    collection = setup_chromadb()

    if args.cleanup:
        # Cleanup mode - remove orphaned entries
        cleanup_database(collection)
    else:
        # Normal mode - populate database
        chunker, tokenizer = setup_chunker()
        embedding_model = setup_embedding_model()
        populate_database(collection, chunker, tokenizer, embedding_model)

    logger.info('Huginn_DB complete')


if __name__ == '__main__':
    main()
