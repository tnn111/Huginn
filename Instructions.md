# A RAG application (Huginn) for a collection of scientific papers

## What I have

I have a folder - Documents - that contains the collection of scientific papers that I want to work with. The collection was produced from PDF documents that I processed with docling using a script (run_docling). Each document in the collection is named either ID.json or ID.md where ID is the base64 encoded representation of the papers Digital Object Identifier (DOI). The run_docling script shows exactly how each was generated.

## What I would like

I would like to build a RAG augmented chatbot with an LLM backend that can make use of the scientific papers in the Documents folder to answer questions. For now, I would like to access the chatbot via a CLI; i.e., I type a question, the chatbot generates an answer and then waits for my next question.

## Component selections I have made

The backend for the chatbot should be Claude with a recent model. I would prefer Claude Sonnet 4.5 for now.

I would like to use IBM Docling. The JSON documents in the Documents folder are the best starting points.

I would like to use the HybridChunker with contextualization provided by the Docling Project.

I would like to start with Chroma DB as the vector store for RAG. I would like the DB to be persisted locally. Save it to Documents_Chroma_DB. I think it would be best to structure the application so that there's code to populate the DB that runs independently of the chatbot. That way I can create and update the DB when needed which is probably not every time I run the chatbot.

## Component selections I have not made

A tokenizer is needed; I have not decided on one.

An embedding solution is needed; I have not decided on a particular one. I think one of the Sentence Transformers would be a good choice.
