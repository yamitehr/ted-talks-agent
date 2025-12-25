import argparse
from logic.ingestion import process_and_ingest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest TED Talks into Pinecone")
    parser.add_argument("--subset", type=int, help="Number of talks to ingest (default: all)")
    
    args = parser.parse_args()
    
    process_and_ingest(subset=args.subset)
