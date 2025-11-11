#!/usr/bin/env python3
"""Huginn - RAG-powered Scientific Paper Chatbot.

Interactive CLI chatbot that answers questions using a collection of scientific papers.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

from huginn_chat_utils import format_citation

# Configuration
CHROMA_DB_DIR = Path('Documents_Chroma_DB')
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
COLLECTION_NAME = 'scientific_papers'
TOP_K = 5  # Number of chunks to retrieve
CLAUDE_MODEL = 'claude-sonnet-4-5-20250929'

# Setup logging
logging.basicConfig(
    level = logging.WARNING,  # Reduce noise in CLI
    format = '%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class HuginnChatbot:
    """RAG-powered chatbot for scientific papers."""

    def __init__(self):
        """Initialize Huginn chatbot."""
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logger.error('ANTHROPIC_API_KEY not found in environment')
            sys.exit(1)

        # Initialize components
        logger.info('Initializing Huginn...')
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = self._load_chromadb()
        self.anthropic = Anthropic(api_key = api_key)

        # Conversation history
        self.conversation_history = []

        logger.info('Huginn ready!')

    def _load_chromadb(self) -> chromadb.Collection:
        """Load ChromaDB collection.

        Returns:
            ChromaDB collection

        Raises:
            SystemExit: If database not found or empty
        """
        if not CHROMA_DB_DIR.exists():
            logger.error(f'Database not found at {CHROMA_DB_DIR}')
            logger.error('Please run Huginn_DB.py first to populate the database')
            sys.exit(1)

        client = chromadb.PersistentClient(
            path = str(CHROMA_DB_DIR),
            settings = Settings(anonymized_telemetry = False)
        )

        try:
            collection = client.get_collection(name = COLLECTION_NAME)
        except Exception as e:
            logger.error(f'Collection "{COLLECTION_NAME}" not found: {e}')
            logger.error('Please run Huginn_DB.py first to populate the database')
            sys.exit(1)

        count = collection.count()
        if count == 0:
            logger.error('Database is empty')
            logger.error('Please run Huginn_DB.py first to populate the database')
            sys.exit(1)

        logger.info(f'Loaded collection with {count} chunks')
        return collection

    def retrieve_context(self, query: str) -> tuple[str, list[dict]]:
        """Retrieve relevant chunks for a query.

        Args:
            query: User's question

        Returns:
            Tuple of (formatted_context, metadata_list)
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy = True
        )

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings = [query_embedding.tolist()],
            n_results = TOP_K,
            include = ['documents', 'metadatas', 'distances']
        )

        # Format context with citations
        context_parts = []
        metadatas = results['metadatas'][0]

        for i, (doc, metadata) in enumerate(zip(results['documents'][0], metadatas)):
            # Build citation
            doi = metadata.get('doi', 'Unknown')
            headings = metadata.get('section_headings', '').split(', ') if metadata.get('section_headings') else None
            page_no = metadata.get('page_number')

            citation = format_citation(doi, headings, page_no)

            # Add to context
            context_parts.append(f'[Source {i+1}: {citation}]\n{doc}\n')

        formatted_context = '\n'.join(context_parts)
        return formatted_context, metadatas

    def generate_response(self, query: str, context: str) -> str:
        """Generate response using Claude with retrieved context.

        Args:
            query: User's question
            context: Retrieved context from papers

        Returns:
            Claude's response
        """
        # Build system message
        system_message = """You are Huginn, an AI assistant specialized in answering questions about scientific papers.

You have access to a collection of scientific papers. When answering questions:

1. Base your answers on the provided context from the scientific papers
2. Cite specific papers using the DOI, section, and page number provided in the sources
3. If the context doesn't contain enough information to answer fully, say so
4. Be precise and scientific in your language
5. When referencing findings, always cite the source

Format citations like this: (DOI: 10.xxxx/xxxxx, Section Name, p. X)"""

        # Build user message with context
        user_message = f"""Context from scientific papers:

{context}

---

User question: {query}

Please answer the question based on the context provided above. Include citations to specific papers and sections."""

        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })

        # Call Claude
        response = self.anthropic.messages.create(
            model = CLAUDE_MODEL,
            max_tokens = 4096,
            system = system_message,
            messages = self.conversation_history
        )

        # Extract assistant response
        assistant_message = response.content[0].text

        # Add to conversation history
        self.conversation_history.append({
            'role': 'assistant',
            'content': assistant_message
        })

        return assistant_message

    def ask(self, query: str) -> str:
        """Process a user query and return response.

        Args:
            query: User's question

        Returns:
            Response from Claude with citations
        """
        # Retrieve relevant context
        context, metadatas = self.retrieve_context(query)

        # Generate response
        response = self.generate_response(query, context)

        return response

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        logger.info('Conversation history reset')


def print_welcome():
    """Print welcome message."""
    print('='*70)
    print('  HUGINN - Scientific Paper Chatbot')
    print('='*70)
    print()
    print('Ask questions about your scientific paper collection.')
    print('Type "exit" or "quit" to end the conversation.')
    print('Type "reset" to clear conversation history.')
    print()


def main():
    """Main entry point for Huginn chatbot."""
    print_welcome()

    try:
        chatbot = HuginnChatbot()
    except Exception as e:
        logger.error(f'Failed to initialize Huginn: {e}')
        sys.exit(1)

    print('Huginn is ready! Ask your first question.')
    print()

    # Main conversation loop
    while True:
        try:
            # Get user input
            query = input('You: ').strip()

            if not query:
                continue

            # Handle commands
            if query.lower() in ['exit', 'quit']:
                print()
                print('Goodbye!')
                break

            if query.lower() == 'reset':
                chatbot.reset_conversation()
                print('Conversation history cleared.')
                print()
                continue

            # Process query
            print()
            print('Huginn: ', end = '', flush = True)
            response = chatbot.ask(query)
            print(response)
            print()

        except KeyboardInterrupt:
            print()
            print()
            print('Goodbye!')
            break
        except Exception as e:
            logger.error(f'Error processing query: {e}', exc_info = True)
            print()
            print(f'Sorry, I encountered an error: {e}')
            print()


if __name__ == '__main__':
    main()
