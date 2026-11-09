from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import cv2
import numpy as np

def cosine(a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / den)

class AppearanceEncoder:
    """Lightweight appearance descriptor; uses color + spatial HSV histograms."""
    def encode(self, crop):
        if crop is None or crop.size == 0:
            return None
        crop = cv2.resize(crop, (64, 128))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Upper/lower body descriptors help distinguish similar global colors.
        parts = [hsv[:64], hsv[64:]]
        feats = []
        for part in parts:
            hist = cv2.calcHist([part], [0,1], None, [16,8], [0,180,0,256])
            hist = cv2.normalize(hist, hist).flatten()
            feats.extend(hist.tolist())
        vec = np.asarray(feats, dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

@dataclass
class Track:
    local_id: int
    global_id: int
    bbox: tuple
    confidence: float
    embedding: np.ndarray
    missed: int = 0
    hits: int = 1

class MultiCameraTracker:
    """Per-camera IoU tracker + shared appearance gallery for cross-camera IDs."""
    def __init__(self, similarity_threshold=0.72, max_missed=20, gallery_momentum=0.8):
        self.encoder = AppearanceEncoder()
        self.threshold = similarity_threshold
        self.max_missed = max_missed
        self.momentum = gallery_momentum
        self.tracks = defaultdict(dict)
        self.next_local = defaultdict(lambda: 1)
        self.next_global = 1
        self.gallery = {}  # global_id -> normalized embedding
        self.colors = {}

    @staticmethod
    def iou(a, b):
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        ix1,iy1=max(ax1,bx1),max(ay1,by1)
        ix2,iy2=min(ax2,bx2),min(ay2,by2)
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        area_a=max(0,ax2-ax1)*max(0,ay2-ay1)
        area_b=max(0,bx2-bx1)*max(0,by2-by1)
        return inter/(area_a+area_b-inter+1e-8)

    def _global_match(self, emb, current_camera):
        best_id, best_score = None, -1.0
        for gid, gallery_emb in self.gallery.items():
            score = cosine(emb, gallery_emb)
            if score > best_score:
                best_id, best_score = gid, score
        if best_score >= self.threshold:
            return best_id
        gid = self.next_global
        self.next_global += 1
        self.gallery[gid] = emb.copy()
        self.colors[gid] = tuple(int(x) for x in np.random.default_rng(gid).integers(70,255,3))
        return gid

    def update(self, camera, frame, detections):
        old = self.tracks[camera]
        candidates = []
        for bbox, conf in detections:
            x1,y1,x2,y2 = map(int,bbox)
            crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            emb = self.encoder.encode(crop)
            if emb is not None:
                candidates.append((tuple(map(int,bbox)), float(conf), emb))

        # Greedy same-camera association by IoU.
        pairs = []
        for di,(bbox,conf,emb) in enumerate(candidates):
            for lid,tr in old.items():
                score = self.iou(bbox, tr.bbox)
                if score > 0.05:
                    pairs.append((score,di,lid))
        pairs.sort(reverse=True)
        used_d, used_t, updated = set(), set(), {}
        for score,di,lid in pairs:
            if di in used_d or lid in used_t: continue
            bbox,conf,emb = candidates[di]
            tr = old[lid]
            gid = tr.global_id
            # Update the global appearance prototype conservatively.
            proto = self.gallery.get(gid, emb)
            merged = self.momentum*proto + (1-self.momentum)*emb
            self.gallery[gid] = merged/(np.linalg.norm(merged)+1e-8)
            updated[lid] = Track(lid,gid,bbox,conf,emb,0,tr.hits+1)
            used_d.add(di); used_t.add(lid)

        for di,(bbox,conf,emb) in enumerate(candidates):
            if di in used_d: continue
            gid = self._global_match(emb, camera)
            lid = self.next_local[camera]; self.next_local[camera] += 1
            updated[lid] = Track(lid,gid,bbox,conf,emb)
        for lid,tr in old.items():
            if lid not in used_t:
                tr.missed += 1
                if tr.missed <= self.max_missed:
                    updated[lid] = tr
        self.tracks[camera] = updated
        return list(updated.values())

    def draw(self, frame, camera, tracks):
        out = frame.copy()
        for tr in tracks:
            if tr.missed: continue
            x1,y1,x2,y2 = tr.bbox
            color = self.colors.get(tr.global_id,(0,220,255))
            cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
            label=f"{camera} | PERSON-{tr.global_id:03d} | {tr.confidence:.2f}"
            (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
            y=max(0,y1-th-10)
            cv2.rectangle(out,(x1,y),(x1+tw+8,y+th+8),color,-1)
            cv2.putText(out,label,(x1+4,y+th+3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(out,f"PERSON RE-ID | {camera} | Active: {sum(t.missed==0 for t in tracks)}",
                    (12,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,(30,240,240),2,cv2.LINE_AA)
        return out
