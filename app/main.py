import base64, os, time
from typing import List, Optional
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Codestra Recognition Engine", version="1.0.0")
DETECTOR_MODEL=os.environ["DETECTOR_MODEL"]
RECOGNIZER_MODEL=os.environ["RECOGNIZER_MODEL"]
detector=cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (320,320), 0.9, 0.3, 5000)
recognizer=cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")
REQ=Counter("recognition_requests_total","Recognition requests",["endpoint","status"])
LAT=Histogram("recognition_request_seconds","Recognition request latency",["endpoint"])

class ImageReq(BaseModel):
    image_base64: str
class VerifyReq(BaseModel):
    image_a_base64: Optional[str]=None
    image_b_base64: Optional[str]=None
    embedding_a: Optional[List[float]]=None
    embedding_b: Optional[List[float]]=None
    threshold: float=0.363
class SearchItem(BaseModel):
    subject_id: str
    embedding: List[float]
class SearchReq(BaseModel):
    image_base64: Optional[str]=None
    embedding: Optional[List[float]]=None
    gallery: List[SearchItem]
    threshold: float=0.363
    top_k: int=5

def decode_image(v:str):
    try:
        raw=base64.b64decode(v, validate=True)
        arr=np.frombuffer(raw,np.uint8)
        img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
        if img is None: raise ValueError("decode")
        return img
    except Exception:
        raise HTTPException(400,"invalid image_base64")

def embedding_from_image(img):
    h,w=img.shape[:2]
    detector.setInputSize((w,h))
    _, faces=detector.detect(img)
    if faces is None or len(faces)==0:
        raise HTTPException(422,"no face detected")
    f=max(faces,key=lambda x: x[2]*x[3])
    aligned=recognizer.alignCrop(img,f)
    feat=recognizer.feature(aligned).flatten().astype(np.float32)
    n=np.linalg.norm(feat)
    if n: feat/=n
    return feat, f[:4].tolist()

def cosine(a,b):
    a=np.asarray(a,dtype=np.float32); b=np.asarray(b,dtype=np.float32)
    if a.size != b.size: raise HTTPException(400,"embedding size mismatch")
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if not na or not nb: return -1.0
    return float(np.dot(a,b)/(na*nb))

@app.get("/healthz")
def healthz(): return {"ok":True}
@app.get("/readyz")
def readyz(): return {"ok":os.path.exists(DETECTOR_MODEL) and os.path.exists(RECOGNIZER_MODEL)}
@app.get("/v1/model/status")
def model_status(): return {"provider":"opencv-sface","detector":"yunet-2023mar","recognizer":"sface-2021dec","metric":"cosine","default_threshold":0.363}
@app.get("/v1/runtime/capabilities")
def capabilities(): return {"cpu":True,"gpu":False,"embeddings":True,"verify":True,"search":True,"batch_search":True}
@app.get("/metrics", response_class=PlainTextResponse)
def metrics(): return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)
@app.post("/v1/embeddings")
def embeddings(req:ImageReq):
    ep="embeddings"; t=time.time()
    try:
        emb,bbox=embedding_from_image(decode_image(req.image_base64)); REQ.labels(ep,"ok").inc()
        return {"embedding":emb.tolist(),"dimension":int(emb.size),"bbox":bbox}
    except Exception:
        REQ.labels(ep,"error").inc(); raise
    finally: LAT.labels(ep).observe(time.time()-t)
@app.post("/v1/verify")
def verify(req:VerifyReq):
    a=req.embedding_a; b=req.embedding_b
    if a is None:
        if not req.image_a_base64: raise HTTPException(400,"image_a_base64 or embedding_a required")
        a,_=embedding_from_image(decode_image(req.image_a_base64)); a=a.tolist()
    if b is None:
        if not req.image_b_base64: raise HTTPException(400,"image_b_base64 or embedding_b required")
        b,_=embedding_from_image(decode_image(req.image_b_base64)); b=b.tolist()
    score=cosine(a,b)
    return {"match":score>=req.threshold,"score":score,"threshold":req.threshold}
@app.post("/v1/search")
def search(req:SearchReq):
    q=req.embedding
    if q is None:
        if not req.image_base64: raise HTTPException(400,"image_base64 or embedding required")
        q,_=embedding_from_image(decode_image(req.image_base64)); q=q.tolist()
    hits=[]
    for item in req.gallery:
        s=cosine(q,item.embedding)
        hits.append({"subject_id":item.subject_id,"score":s,"match":s>=req.threshold})
    hits.sort(key=lambda x:x["score"],reverse=True)
    return {"hits":hits[:max(1,min(req.top_k,100))],"threshold":req.threshold}
@app.post("/v1/batch/search")
def batch_search(reqs:List[SearchReq]): return [search(r) for r in reqs]
