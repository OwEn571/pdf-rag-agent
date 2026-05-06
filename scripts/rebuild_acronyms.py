"""Fast acronym-only rebuild — pypdf text only, no hi_res, ~2 min."""

import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.services.retrieval.indexing import IngestionService
from app.services.retrieval.pdf_extractor import PDFExtractor
from app.services.library.zotero_sqlite import ZoteroSQLiteReader
from app.services.retrieval.vector_index import CollectionVectorIndex

settings = get_settings()

# Load records
reader = ZoteroSQLiteReader(settings)
records = reader.read_records()
print(f"Loaded {len(records)} records")

# Fast pypdf-only (skip hi_res!)
extractor = PDFExtractor(settings, prefer_unstructured=False)
service = IngestionService(settings)

paper_docs = []
acronym_stats = {"total_papers": 0, "with_map": 0, "total_pairs": 0}

for i, record in enumerate(records):
    if not record.file_exists:
        continue
    try:
        pages = extractor.extract_pages(record.file_path)
    except Exception:
        continue

    # Rebuild paper card with new acronym extraction
    doc, generated = service._build_paper_card(record=record, pages=pages)

    # Check acronym_map
    raw_map = doc.metadata.get("acronym_map", "")
    acr_map = {}
    if raw_map:
        try:
            acr_map = json.loads(raw_map)
        except (json.JSONDecodeError, TypeError):
            pass

    acronym_stats["total_papers"] += 1
    if acr_map:
        acronym_stats["with_map"] += 1
        acronym_stats["total_pairs"] += len(acr_map)

    if i < 5:
        bac = doc.metadata.get("body_acronyms", "")
        real = [a for a in bac.split("||") if a and a.upper() == a and 3 <= len(a) <= 10]
        print(f"  [{record.title[:55]}] acronyms={len(bac.split('||'))} short={len(real)} map={len(acr_map)}")
        if acr_map:
            print(f"    KV: {dict(list(acr_map.items())[:5])}")

    paper_docs.append(doc)

    if (i + 1) % 30 == 0:
        print(f"  {i+1}/{len(records)} done...")

print(f"\nStats: {acronym_stats['total_papers']} papers, {acronym_stats['with_map']} with acronym_map, {acronym_stats['total_pairs']} total KV pairs")

# Write JSONL
print(f"\nWriting {len(paper_docs)} paper cards to {settings.paper_store_path}...")
with open(settings.paper_store_path, "w", encoding="utf-8") as f:
    for doc in paper_docs:
        f.write(json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False) + "\n")

# Re-upsert to Milvus
print("Upserting to Milvus paper collection...")
paper_index = CollectionVectorIndex(settings, collection_name=settings.milvus_paper_collection)
# Clear old vectors
from pymilvus import MilvusClient, connections
try:
    client = MilvusClient(uri=settings.milvus_uri)
    client.drop_collection(settings.milvus_paper_collection)
    print("  Old collection dropped")
except Exception:
    pass
paper_index = CollectionVectorIndex(settings, collection_name=settings.milvus_paper_collection)
paper_index.upsert_documents(paper_docs, batch_size=128)
paper_index.close()
print("  Paper collection re-upserted")

print("\nDone!")
