#!/usr/bin/env python3
"""Huginn Database Management Script.

Populates ChromaDB vector store with chunked scientific papers from Documents folder.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Iterator
import sys

from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from huginn_utils import (
    decode_doi_from_filename,
    extract_document_title,
    extract_authors,
    get_page_number,
)

# Suppress deprecation warnings from docling about annotations -> meta migration
warnings.filterwarnings('ignore', message='Migrating deprecated')

# Configuration
DOCUMENTS_DIR = Path('Documents')
CHROMA_DB_DIR = Path('Documents_Chroma_DB')
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
MAX_TOKENS = 512
COLLECTION_NAME = 'scientific_papers'

# Setup logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)
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
    tokenizer: HuggingFaceTokenizer
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
            headings = chunk.meta.headings if chunk.meta.headings else []

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
    # Find all JSON files
    json_files = list(DOCUMENTS_DIR.glob('*.json'))
    logger.info(f'Found {len(json_files)} JSON files to process')

    if len(json_files) == 0:
        logger.warning(f'No JSON files found in {DOCUMENTS_DIR}')
        return

    total_chunks = 0
    processed_docs = 0
    failed_docs = 0

    for json_path in json_files:
        try:
            # Collect chunks from this document
            batch_texts = []
            batch_metadatas = []
            batch_ids = []

            for chunk_text, metadata, chunk_id in process_document(json_path, chunker, tokenizer):
                batch_texts.append(chunk_text)
                batch_metadatas.append(metadata)
                batch_ids.append(chunk_id)

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

        except Exception as e:
            failed_docs += 1
            logger.error(f'Failed to process {json_path.name}: {e}')
            continue

    # Summary
    logger.info('='*60)
    logger.info('Database population complete!')
    logger.info(f'  Successfully processed: {processed_docs} documents')
    logger.info(f'  Failed: {failed_docs} documents')
    logger.info(f'  Total chunks added: {total_chunks}')
    logger.info(f'  Final collection count: {collection.count()}')
    logger.info('='*60)


def main():
    """Main entry point for Huginn_DB."""
    logger.info('Starting Huginn Database Management')
    logger.info('='*60)

    # Check if Documents directory exists
    if not DOCUMENTS_DIR.exists():
        logger.error(f'Documents directory not found: {DOCUMENTS_DIR}')
        sys.exit(1)

    # Setup components
    chunker, tokenizer = setup_chunker()
    embedding_model = setup_embedding_model()
    collection = setup_chromadb()

    # Populate database
    populate_database(collection, chunker, tokenizer, embedding_model)

    logger.info('Huginn_DB complete')


if __name__ == '__main__':
    main()
