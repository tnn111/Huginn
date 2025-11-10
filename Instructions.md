# A RAG application (Huginn) for a collection of scientific papers

## What I have

I have a folder - Documents - that contains the collection of scientific papers that I want to work with. The collection was produced from PDF documents that I processed with docling using a script (run_docling). Each document in the collection is named either ID.json or ID.md where ID is the base64 encoded representation of the papers Digital Object Identifier (DOI). The run_docling script shows exactly how each was generated.

The Documents folder only contains ~300 papers at the moment, but you should be prepared to handle 1,000.

## What I would like

I would like to build a RAG augmented chatbot with an LLM backend that can make use of the scientific papers in the Documents folder to answer questions. For now, I would like to access the chatbot via a CLI; i.e., I type a question, the chatbot generates an answer and then waits for my next question.

## Component selections I have made

The backend for the chatbot should be Claude with a recent model. I would prefer Claude Sonnet 4.5 for now. I have an Anthropic API key.

I would like to use IBM Docling. The JSON documents in the Documents folder are the best starting points.

I would like to use the HybridChunker provided by the Docling Project. To the extent possible, the document structure should be preserved. Instead of overlap, I want to use the contextualization provided by the chunker. A target of 512 tokens is good. Metadata should be stored with chunks.

I would like to start with Chroma DB as the vector store for RAG. I would like the DB to be persisted locally. Save it to Documents_Chroma_DB. 

I would like code to manage the DB independently of the chatbot. That way I can create and update the DB when needed which is probably not every time I run the chatbot.

Each query to the chatbot should retrieve the top 5 matches. Claude should reference the specific paper and section when answering. I do not want to use reranking for now, but I may wish to add it later.

## Component selections I have not made

A tokenizer is needed; I have not decided on one.

A tokenizer and an embedding solution are needed; I have not decided on a particular one. I think one of the Sentence Transformers along with a matching tokenizer would be a good choice. Let's start with all-mpnet-base-v2 for the embedding model.