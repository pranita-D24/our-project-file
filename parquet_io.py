# parquet_io.py — Stage 3 Optimized Bulk Storage
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════
# DATA SCHEMA DEFINITION (Strict)
# ══════════════════════════════════════

ELEMENT_SCHEMA = pa.schema([
    ('id', pa.string()),
    ('drawing_id', pa.string()),
    ('prefix', pa.string()), # Partition key 1
    ('revision', pa.string()), # Partition key 2
    ('type', pa.string()),
    ('bbox_x', pa.int32()),
    ('bbox_y', pa.int32()),
    ('bbox_w', pa.int32()),
    ('bbox_h', pa.int32()),
    ('centroid_x', pa.float64()),
    ('centroid_y', pa.float64()),
    ('text', pa.string()),
    ('value', pa.float64()),
    ('confidence', pa.float64()),
    ('metadata_json', pa.string()),
    ('processed_at', pa.timestamp('ms'))
])

# ══════════════════════════════════════
# PARQUET MANAGER
# ══════════════════════════════════════

class ParquetStore:
    def __init__(self, base_path: str = "storage/parquet_data"):
        self.base_path = base_path
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            logger.info(f"Created base parquet directory: {base_path}")

    def save_elements(self, elements: List[Dict[str, Any]]):
        """APPLY FIX: Partition by drawing prefix + revision"""
        if not elements: return
        df = pd.DataFrame(elements)
        if 'drawing_id' in df.columns:
            df['prefix'] = df['drawing_id'].str[:4]
        if 'revision' not in df.columns:
            df['revision'] = 'V1'
            
        try:
            # APPLY FIX: Ensure all schema columns exist (with nulls for missing)
            col_names = [field.name for field in ELEMENT_SCHEMA]
            df = df.reindex(columns=col_names)
            
            table = pa.Table.from_pandas(df, schema=ELEMENT_SCHEMA)
            pq.write_to_dataset(
                table, root_path=self.base_path,
                partition_cols=['prefix', 'revision'],
                use_legacy_dataset=False, compression='snappy'
            )
        except Exception as e:
            logger.error(f"Parquet save failed: {e}")

    def load_elements(self, drawing_id: str = None, revision: str = None) -> pd.DataFrame:
        """APPLY FIX: Use filter pushdown for Parquet reads"""
        filters = []
        if drawing_id:
            filters.append(('prefix', '==', drawing_id[:4]))
            filters.append(('drawing_id', '==', drawing_id))
        if revision:
            filters.append(('revision', '==', revision))

        try:
            # Filter pushdown ensures only relevant data is read from disk
            table = pq.read_table(self.base_path, filters=filters if filters else None, schema=ELEMENT_SCHEMA)
            return table.to_pandas()
        except:
            return pd.DataFrame()

    def stream_all_data(self, chunk_size=10000):
        """
        Generator for streaming reads of the entire dataset.
        Prevents memory exhaustion on 270k+ diagrams.
        """
        dataset = pq.ParquetDataset(self.base_path, use_legacy_dataset=False)
        for fragment in dataset.fragments:
            table = fragment.to_table(schema=ELEMENT_SCHEMA)
            # Process in chunks if table is very large
            num_rows = table.num_rows
            for i in range(0, num_rows, chunk_size):
                yield table.slice(i, chunk_size).to_pandas()
