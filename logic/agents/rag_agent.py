import os
from typing import List, Tuple, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import config

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEXT = """You are a TED Talk assistant that answers questions strictly and
only based on the TED dataset context provided to you (metadata
and transcript passages). You must not use any external
knowledge, the open internet, or information that is not explicitly
contained in the retrieved context. If the answer cannot be
determined from the provided context, respond: “I don’t know
based on the provided TED data.” Always explain your answer
using the given context, quoting or paraphrasing the relevant
transcript or metadata when helpful.

Response Style Guidelines:
1. If the user asks for a recommendation, pick the single most relevant talk and justify your choice with specific details from the context. Do not offer a menu of options unless explicitly asked for a list.
2. Do not reference the retrieval process itself (e.g., avoid phrases like "Based on the provided chunks" or "I found 3 talks"). Speak directly as an assistant knowledgeable about these specific talks."""

class RAGAgent:
    def __init__(self):
        self.embeddings = self._get_embeddings()
        self.llm = self._get_llm()
        self.vectorstore = PineconeVectorStore(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=self.embeddings,
            pinecone_api_key=config.PINECONE_API_KEY
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_TEXT),
            ("user", "Context:\n{context}\n\nQuestion: {question}")
        ])

    def _get_llm(self):
        return ChatOpenAI(model="RPRTHPB-gpt-5-mini", temperature=1, base_url=config.OPENAI_BASE_URL)

    def _get_embeddings(self):
        return OpenAIEmbeddings(model="RPRTHPB-text-embedding-3-small", base_url=config.OPENAI_BASE_URL)

    async def aask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Asynchronous retrieval and generation.
        Returns a dictionary with response text, context documents with scores, and prompt messages.
        """
        logger.info(f"Received question: {question}")
        
        # 1. Retrieve with scores
        logger.info(f"Retrieving top {top_k} documents from Pinecone...")
        docs_and_scores = self.vectorstore.similarity_search_with_score(question, k=top_k)
        logger.info(f"Retrieved {len(docs_and_scores)} documents.")
        
        # Format context
        # We explicitly include metadata here so the LLM knows the source of every chunk
        formatted_context = "\n\n".join([
            f"Title: {d.metadata.get('title', 'Unknown')}\n"
            f"Speaker: {d.metadata.get('speaker', 'Unknown')}\n"
            f"Content: {d.page_content}"
            for d, _ in docs_and_scores
        ])
        
        # 2. Format Prompt
        messages = self.prompt_template.format_messages(context=formatted_context, question=question)
        system_msg_content = messages[0].content
        user_msg_content = messages[1].content
        
        # 3. Generate
        logger.info("Invoking LLM for generation...")
        chain = self.llm | StrOutputParser()
        response_text = await chain.ainvoke(messages)
        logger.info("LLM generation complete.")
        
        return {
            "response_text": response_text,
            "docs_and_scores": docs_and_scores,
            "system_msg": system_msg_content,
            "user_msg": user_msg_content
        }

# Singleton instance
rag_agent = RAGAgent()