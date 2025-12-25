import pandas as pd
import time
from typing import List, Optional, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

import config

def get_embeddings():
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAIEmbeddings(model="RPRTHPB-text-embedding-3-small", base_url=config.OPENAI_BASE_URL)

def prepare_docs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Fill NaNs
    df = df.fillna("")
    
    docs = []
    # iterating manually to control the content format
    for _, row in df.iterrows():
        content = (
            f"Title: {row['title']}\n"
            f"Speaker: {row['speaker_1']}\n"
            f"Occupations: {row['occupations']}\n"
            f"About Speakers: {row['about_speakers']}\n"
            f"Published Date: {row['published_date']}\n"
            f"Topics: {row['topics']}\n"
            f"Description: {row['description']}\n"
            f"Transcript: {row['transcript']}"
        )
        
        metadata = {
            "talk_id": str(row['talk_id']),
            "title": row['title'],
            "speaker": row['speaker_1'],
            "url": row['url'],
            "topics": row['topics']
        }
        
        docs.append({"text": content, "metadata": metadata})
        
    return docs

def process_and_ingest(subset: Optional[int] = None, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
    print(f"Starting ingestion. Subset: {subset}")
    
    # 1. Load Data
    print(f"Loading data from {config.DATA_PATH}...")
    try:
        df = pd.read_csv(config.DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {config.DATA_PATH}")
        return
    
    if subset:
        df = df.head(subset)
        print(f"Subsetting to first {subset} rows.")
        
    print(f"Loaded {len(df)} talks.")
    
    # 2. Prepare Documents
    raw_docs_data = prepare_docs(df)
    
    # 3. Split
    print(f"Splitting with chunk_size={chunk_size} tokens, overlap={chunk_overlap} tokens...")
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    texts = [d["text"] for d in raw_docs_data]
    metadatas = [d["metadata"] for d in raw_docs_data]
    
    if not texts:
        print("No documents to split.")
        return

    splits = text_splitter.create_documents(texts, metadatas=metadatas)
    print(f"Created {len(splits)} chunks.")
    
    # 4. Initialize Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index_name = config.PINECONE_INDEX_NAME
    
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    existing_ids = set()
    
    if index_name not in existing_indexes:
        print(f"Index '{index_name}' does not exist. Creating...")
        
        dims = 1536
            
        try:
            pc.create_index(
                name=index_name,
                dimension=dims,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index '{index_name}' created.")
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        except Exception as e:
            print(f"Error creating index: {e}")
            return
    else:
        # Index exists, let's fetch existing IDs to avoid re-embedding
        print(f"Index '{index_name}' exists. Checking for existing vectors...")
        try:
            index = pc.Index(index_name)
            # List IDs using pagination
            for ids in index.list():
                existing_ids.update(ids)
            print(f"Found {len(existing_ids)} existing chunks (vectors) in index.")
        except Exception as e:
            print(f"Warning: Could not fetch existing IDs ({e}). Proceeding with caution.")

    # 5. Filter Data (Optimization)
    # The IDs in Pinecone are likely chunk IDs (e.g., "talk_id_0", "talk_id_1"), 
    # but we need to know if the TALK itself is processed.
    # Our simple logic: if we find ANY chunk for a talk_id, we assume the talk is processed.
    # NOTE: This assumes the previous ingestion was successful for that talk.
    
    processed_talk_ids = set()
    for vid in existing_ids:
        # Assuming ID format might be flexible, but usually we don't set custom IDs in from_documents 
        # unless we specify them. LangChain usually generates UUIDs by default unless we handle it.
        # WAIT: LangChain's PineconeVectorStore generates random UUIDs for chunks by default.
        # This makes it hard to dedup based on ID unless we used deterministic IDs.
        pass
        
    # CRITICAL: LangChain by default generates random UUIDs. We can't easily check duplication 
    # unless we check metadata. Pinecone's list() returns IDs, not metadata.
    # Querying metadata for every talk is expensive.
    
    # IMPROVEMENT: We will change the ID generation to be deterministic: f"{talk_id}_{chunk_index}"
    # This allows us to check if specific chunks exist.
    
    # Let's adjust the splitting logic first to generate IDs, then filter.

    # 2. Prepare Documents
    raw_docs_data = prepare_docs(df)
    
    # 3. Split
    print(f"Splitting with chunk_size={chunk_size} tokens, overlap={chunk_overlap} tokens...")
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    texts = [d["text"] for d in raw_docs_data]
    metadatas = [d["metadata"] for d in raw_docs_data]
    
    if not texts:
        print("No documents to split.")
        return

    splits = text_splitter.create_documents(texts, metadatas=metadatas)
    
    # Assign Deterministic IDs to Chunks
    # Format: talk_id_chunk_index
    final_docs = []
    final_ids = []
    
    # We need to group splits by talk_id to assign sequential indices
    from collections import defaultdict
    splits_by_talk = defaultdict(list)
    for s in splits:
        t_id = s.metadata.get("talk_id")
        splits_by_talk[t_id].append(s)
        
    skipped_count = 0
    
    for t_id, talk_splits in splits_by_talk.items():
        for i, doc in enumerate(talk_splits):
            chunk_id = f"{t_id}_{i}"
            
            if chunk_id in existing_ids:
                skipped_count += 1
                continue
            
            final_docs.append(doc)
            final_ids.append(chunk_id)
            
    print(f"Total chunks generated: {len(splits)}")
    print(f"Skipping {skipped_count} chunks that are already indexed.")
    print(f"New chunks to embed: {len(final_docs)}")
    
    if not final_docs:
        print("No new data to ingest.")
        return

    # 6. Embed and Upsert
    embeddings = get_embeddings()
    
    print("Upserting to Pinecone...")
    # We use add_documents instead of from_documents to pass specific IDs
    vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)
    vectorstore.add_documents(documents=final_docs, ids=final_ids)
    
    print("Ingestion complete.")