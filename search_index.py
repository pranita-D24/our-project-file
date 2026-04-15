# search_index.py — Stage 3 Search & Metadata Indexing
import logging
import json
from typing import List, Dict, Any, Optional
try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    Elasticsearch = None

logger = logging.getLogger(__name__)

# ══════════════════════════════════════
# ELASTICSEARCH MAPPING (Production)
# ══════════════════════════════════════

ELEMENT_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "drawing_id": {"type": "keyword"},
            "type": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "standard"},
            # STAGE 3 Fix: Explicit numeric field mapping
            "dimension_value": {"type": "double"},
            "metadata": {"type": "object"},
            "processed_at": {"type": "date"}
        }
    }
}

# ══════════════════════════════════════
# SEARCH INDEX MANAGER
# ══════════════════════════════════════

class SearchIndexManager:
    def __init__(self, hosts: List[str] = ["http://localhost:9200"]):
        self.es = None
        if Elasticsearch:
            try:
                self.es = Elasticsearch(hosts)
                if not self.es.indices.exists(index="elements"):
                    # APPLY FIX: Numeric field mapping
                    self.es.indices.create(index="elements", body=ELEMENT_MAPPING)
                logger.info(f"Connected to Elasticsearch")
            except:
                self.es = None
        self.mock_store = []

    def index_elements_bulk(self, elements: List[Dict[str, Any]]):
        """
        Indexes elements into ES (production) or mock store (dev).
        """
        if self.es:
            try:
                actions = [
                    {
                        "_index": "elements",
                        "_id": e['id'],
                        "_source": {
                            "drawing_id": e['drawing_id'],
                            "type": e['type'],
                            "text": e.get('text', ''),
                            "dimension_value": e.get('value'),
                            "metadata": e.get('metadata', {}),
                            "processed_at": e.get('processed_at')
                        }
                    }
                    for e in elements
                ]
                helpers.bulk(self.es, actions)
                logger.info(f"Bulk indexed {len(elements)} items into Elasticsearch.")
            except Exception as e:
                logger.error(f"ES Bulk indexing failed: {e}")
        else:
            # Fallback mock indexing
            self.mock_store.extend(elements)
            logger.info(f"Mock indexed {len(elements)} items locally.")

    def search_text(self, query: str, drawing_id: Optional[str] = None) -> List[Dict]:
        """
        Fuzzy text search across drawing annotations and notes.
        """
        if self.es:
            body = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"text": {"query": query, "fuzziness": "AUTO"}}}
                        ]
                    }
                }
            }
            if drawing_id:
                body["query"]["bool"]["filter"] = [{"term": {"drawing_id": drawing_id}}]
                
            res = self.es.search(index="elements", body=body)
            return [hit["_source"] for hit in res["hits"]["hits"]]
        else:
            # Mock fuzzy search: simple substring check
            query = query.lower()
            return [e for e in self.mock_store if query in str(e.get('text', '')).lower()]

    def search_by_value_range(self, min_val: float, max_val: float) -> List[Dict]:
        """
        Search for elements within a numeric range (e.g. dimensions).
        """
        if self.es:
            body = {
                "query": {
                    "range": {
                        "dimension_value": {
                            "gte": min_val,
                            "lte": max_val
                        }
                    }
                }
            }
            res = self.es.search(index="elements", body=body)
            return [hit["_source"] for hit in res["hits"]["hits"]]
        else:
            return [e for e in self.mock_store if e.get('value') is not None and min_val <= e['value'] <= max_val]
