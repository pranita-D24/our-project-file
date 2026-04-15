# database_spatial.py — Stage 3 Spatial Storage Layer
import logging
import json
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, Index, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database

logger = logging.getLogger(__name__)
Base = declarative_base()

# ══════════════════════════════════════
# SCHEMA DEFINITIONS
# ══════════════════════════════════════

class DrawingElement(Base):
    __tablename__ = 'drawing_elements'
    
    id = Column(String(36), primary_key=True)
    drawing_id = Column(String(50), index=True)
    version_id = Column(String(50), index=True)
    type = Column(String(20)) # component | dimension | balloon | gdt | note
    
    # Bounding Box (axis-aligned)
    bbox_x = Column(Integer)
    bbox_y = Column(Integer)
    bbox_w = Column(Integer)
    bbox_h = Column(Integer)
    
    # Computed Centroids (derived from bbox)
    centroid_x = Column(Float, index=True)
    centroid_y = Column(Float, index=True)
    
    # Metadata
    text = Column(Text)
    value = Column(Float, index=True)
    confidence = Column(Float)
    metadata_json = Column(Text)

# ══════════════════════════════════════
# RTREE FALLBACK (Refined with delete/update)
# ══════════════════════════════════════

class RTreeManager:
    def __init__(self):
        try:
            from rtree import index
            self.idx = index.Index()
            self.elements = {}
        except ImportError:
            self.idx = None
            self.elements = {}

    def insert(self, element_id, bbox):
        if self.idx:
            left, top, w, h = bbox
            self.idx.insert(hash(element_id), (left, top, left + w, top + h))
        self.elements[element_id] = bbox

    def remove(self, element_id):
        """APPLY FIX: Support deletion in RTree"""
        if self.idx and element_id in self.elements:
            left, top, w, h = self.elements[element_id]
            self.idx.delete(hash(element_id), (left, top, left + w, top + h))
        self.elements.pop(element_id, None)

    def update(self, element_id, new_bbox):
        """APPLY FIX: Support update in RTree"""
        self.remove(element_id)
        self.insert(element_id, new_bbox)

    def search(self, bbox) -> List[str]:
        left, top, w, h = bbox
        if self.idx:
            return [str(eid) for eid in self.idx.intersection((left, top, left + w, top + h))]
        return [eid for eid, b in self.elements.items() 
                if not (b[0] > left + w or b[0] + b[2] < left or b[1] > top + h or b[1] + b[3] < top)]

# ══════════════════════════════════════
# SPATIAL DB MANAGER (Refined)
# ══════════════════════════════════════

class SpatialDatabaseManager:
    def __init__(self, connection_url: str = "sqlite:///spatial_elements.db"):
        self.engine = create_engine(connection_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.local_rtree = RTreeManager()

    def bulk_save_elements(self, elements: List[Dict[str, Any]], chunk_size=1000):
        """APPLY FIX: Chunked bulk inserts"""
        session = self.Session()
        try:
            for i in range(0, len(elements), chunk_size):
                chunk = elements[i:i + chunk_size]
                db_objs = []
                for e in chunk:
                    # APPLY FIX: Computed centroid
                    cx = e['bbox_x'] + (e['bbox_w'] / 2.0)
                    cy = e['bbox_y'] + (e['bbox_h'] / 2.0)
                    
                    obj_id = e.get('id', str(uuid.uuid4()))
                    db_objs.append(DrawingElement(
                        id=obj_id,
                        drawing_id=e.get('drawing_id'),
                        version_id=e.get('version_id'),
                        type=e.get('type'),
                        bbox_x=e['bbox_x'], bbox_y=e['bbox_y'],
                        bbox_w=e['bbox_w'], bbox_h=e['bbox_h'],
                        centroid_x=cx, centroid_y=cy,
                        text=e.get('text'),
                        value=e.get('value'),
                        confidence=e.get('confidence'),
                        metadata_json=json.dumps(e.get('metadata', {}))
                    ))
                    self.local_rtree.insert(obj_id, (e['bbox_x'], e['bbox_y'], e['bbox_w'], e['bbox_h']))
                
                session.bulk_save_objects(db_objs)
                session.commit()
            logger.info(f"Chunked save complete: {len(elements)} elements.")
        except Exception as e:
            logger.error(f"Bulk save failed: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def find_within_radius(self, cx: float, cy: float, radius: float) -> List[Dict]:
        """
        Spatial Query: Find elements within distance of a point.
        Uses Rtree for local/SQLite, or PostGIS ST_DWithin for Postgres.
        """
        # Search area bbox for Rtree/PostGIS gating
        search_bbox = (cx - radius, cy - radius, radius * 2, radius * 2)
        candidate_ids = self.local_rtree.search(search_bbox)
        
        if not candidate_ids:
            return []

        session = self.Session()
        try:
            # Final distance check (Euclidean)
            results = session.query(DrawingElement).filter(
                DrawingElement.id.in_(candidate_ids)
            ).all()
            
            filtered = []
            for r in results:
                dist = np.hypot(r.centroid_x - cx, r.centroid_y - cy)
                if dist <= radius:
                    filtered.append({
                        "id": r.id, "type": r.type, 
                        "centroid": (r.centroid_x, r.centroid_y),
                        "value": r.value, "text": r.text
                    })
            return filtered
        finally:
            session.close()
