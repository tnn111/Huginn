# A RAG application (Huginn) for a collection of scientific papers

## What I have

I have a folder - Documents - that contains the collection of scientific papers that I want to work with. The collection was produced from PDF documents that I processed with docling using a script (run_docling). Each document in the collection is named either ID.json or ID.md where ID is the base64 encoded representation of the papers Digital Object Identifier (DOI). The run_docling script shows exactly how each was generated.

The Documents folder only contains ~300 papers at the moment, but you should be prepared to handle 1,000.

## What I would like

I would like to build a RAG augmented chatbot with an LLM backend that can make use of the scientific papers in the Documents folder to answer questions. For now, I would like to access the chatbot via a CLI; i.e., I type a question, the chatbot generates an answer and then waits for my next question.

## Component selections I have made

The backend for the chatbot should be Claude with a recent model. I would prefer Claude Sonnet 4.5 for now. I have an Anthropic API key. It is stored in a .env file as ANTHROPIC_API_KEY.

I would like to use IBM Docling. The JSON documents in the Documents folder are the best starting points.

I would like to use the HybridChunker provided by the Docling Project. To the extent possible, the document structure should be preserved. Instead of overlap, I want to use the contextualization provided by the chunker. A target of 512 tokens is good.

Metadata should be stored with chunks. The HybridChunker automatically provides rich metadata including:
- Section headings (via chunk.meta.headings)
- Captions (via chunk.meta.captions)
- Page numbers (via chunk.meta.doc_items[].prov[].page_no)
- Origin information (filename, mimetype, binary_hash via chunk.meta.origin)

Additional custom metadata to extract and store:
- DOI: Extract from the JSON filename by decoding the base64-encoded portion (e.g., "MTAuMjE3NjkvQmlvUHJvdG9jLjE4MTg" decodes to "10.21769/BioProtoc.1818")
- Chunk position: Sequential index of the chunk within the document (0, 1, 2...)
- Document title: Extract from the document (typically the first section_header in the document)
- Authors: Attempt to extract author names from the document text (usually near the beginning)

A tokenizer and an embedding solution are needed. I think one of the Sentence Transformers along with a matching tokenizer would be a good choice. Let's start with all-mpnet-base-v2 for the embedding model and its corresponding tokenizer.

I would like to start with Chroma DB as the vector store for RAG. I would like the DB to be persisted locally. Save it to Documents_Chroma_DB. 

Each query to the chatbot should retrieve the top 5 matches. Claude should embed citations in the text when answering. The chatbot should decode the base64 and use the DOI to display. I do not want to use reranking for now, but I may wish to add it later. For the first version, the chatbot should maintain context across questions.

## DB choice and management

I would like to start with Chroma DB as the vector store for RAG. I would like the DB to be persisted locally. Save it to Documents_Chroma_DB. DB management should be handled separately from the chatbot. That is, it should be a separate piece of code. The DB management should handle initial population with all JSON files in the Documents folder. The first version need not do more than that. If a JSON file is malformed or a paper fails to process, it should be logged but the processing should continue.

## Technical Implementation Details

Based on analysis of the Docling document structure:

### Document Structure
- Each DoclingDocument JSON contains: `schema_name`, `version`, `name`, `origin`, `body`, `furniture`, `groups`, `texts`, `pictures`, `tables`, `pages`
- Text items have: `label` (e.g., "text", "section_header"), `text` content, `prov` (provenance with page_no and bbox)
- Document hierarchy is maintained through parent/child references using JSON pointers
- Pages are stored as a dictionary with page numbers as keys

### Chunking Process
1. Load DoclingDocument from JSON using `DoclingDocument.model_validate(json_dict)`
2. Create HuggingFaceTokenizer with `sentence-transformers/all-mpnet-base-v2` tokenizer, max_tokens=512
3. Create HybridChunker with the tokenizer
4. Generate chunks using `chunker.chunk(doc)`
5. For embedding, use `chunker.contextualize(chunk)` which adds relevant context (headings, etc.)

### Chunk Metadata Extraction
Each chunk (DocChunk object) provides:
- `chunk.text`: Raw chunk text
- `chunk.meta.headings`: List of section headings (hierarchical context)
- `chunk.meta.captions`: Any associated captions
- `chunk.meta.origin.filename`: Original PDF filename
- `chunk.meta.doc_items[0].prov[0].page_no`: Page number (if available)

Custom metadata to add:
- Extract DOI from JSON filename using base64 decode
- Track chunk index/position
- Extract document title from first section_header or early text
- Attempt to extract authors from early document text

### Citation Format
When Claude responds, citations should display the decoded DOI, section heading, and page number, e.g.:
"According to the study (DOI: 10.21769/BioProtoc.1818, Materials and Reagents section, p. 1)..."