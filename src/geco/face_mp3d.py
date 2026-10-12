"""Photo -> person-specific 3D face using the MediaPipe 468-landmark network (no torch needed).

Pipeline: YuNet box -> MediaPipe face_landmark.tflite (468 x,y,z) -> similarity (Umeyama) fit to the canonical
468-vertex topology -> pose-normalised, person-specific mesh -> skull vault follows the face deformation ->
photo-textured z-buffer renderer, .obj export, and a per-vertex residual ("anomaly") map.

Model files: run  python scripts/extract_mediapipe_models.py  once (extracts them from frontend/public/mediapipe).
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from .face_align import extract_face_bbox_and_crop

_HERE=Path(__file__).resolve().parent
MODEL_DIR=Path(os.environ.get("GECO_MP_MODELS", _HERE.parent.parent/"models"/"mediapipe"))
_it=None
def landmarks468(rgb, bbox):
    """MediaPipe face_landmark.tflite on a square crop around the YuNet bbox. Returns [468,3] in image px (z in px-scale)."""
    global _it
    from ai_edge_litert.interpreter import Interpreter
    if _it is None:
        _it=Interpreter(model_path=str(MODEL_DIR/'face_landmark.tflite')); _it.allocate_tensors()
    h,w=rgb.shape[:2]; x,y,bw,bh=bbox
    cx,cy=x+bw/2,y+bh/2; s=max(bw,bh)*1.5
    M=np.array([[192/s,0,192/2-cx*192/s],[0,192/s,192/2-cy*192/s]],np.float32)
    crop=cv2.warpAffine(rgb,M,(192,192),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    inp=(crop.astype(np.float32)/255.)[None]
    _it.set_tensor(_it.get_input_details()[0]['index'],inp); _it.invoke()
    out=_it.get_tensor(_it.get_output_details()[0]['index']).reshape(468,3)
    score=float(_it.get_tensor(_it.get_output_details()[1]['index']).ravel()[0])
    Mi=cv2.invertAffineTransform(M)
    xy=out[:,:2]@Mi[:,:2].T+Mi[:,2]
    z=out[:,2]*(s/192)
    return np.c_[xy,z].astype(np.float32), score


D=json.load(open(_HERE/'canonical_head_mesh.json'))
CV=np.array(D['vertices'],np.float32); TRI=np.array(D['triangles'],np.int64)
CRAN=np.zeros(len(CV),bool); CRAN[468:]=True
BACK=CRAN&(CV[:,2]<=-0.02)          # true skull verts
FILL=CRAN&~BACK                      # 3 centroid verts that close eyes+mouth (belong to the face)
CAN468=CV[:468]

def umeyama(src,dst):
    mu_s,mu_d=src.mean(0),dst.mean(0); S=src-mu_s; Dd=dst-mu_d
    U,sig,Vt=np.linalg.svd(Dd.T@S/len(src)); d=np.sign(np.linalg.det(U@Vt)); E=np.diag([1,1,d])
    R=U@E@Vt; s=(sig*np.diag(E)).sum()/S.var(0).sum(); t=mu_d-s*R@mu_s
    return R,s,t

def adjacency(n,tri):
    A=[set() for _ in range(n)]
    for a,b,c in tri:
        A[a]|={b,c}; A[b]|={a,c}; A[c]|={a,b}
    return [np.array(sorted(x)) for x in A]
ADJ=adjacency(len(CV),TRI)

def _fill_holes(V,tri):
    from collections import defaultdict
    cnt=defaultdict(int); nxt={}
    for a,b,c in tri:
        for u,v in ((a,b),(b,c),(c,a)): cnt[(min(u,v),max(u,v))]+=1
    bedges=[(u,v) for a,b,c in tri for u,v in ((a,b),(b,c),(c,a)) if cnt[(min(u,v),max(u,v))]==1]
    nb=defaultdict(list)
    for u,v in bedges: nb[u].append(v)
    seen=set(); loops=[]
    for u,v in bedges:
        if (u,v) in seen: continue
        loop=[u]; cur=v; seen.add((u,v))
        while cur!=u and len(loop)<400:
            loop.append(cur); nx=[w for w in nb[cur] if (cur,w) not in seen]
            if not nx: break
            seen.add((cur,nx[0])); cur=nx[0]
        loops.append(loop)
    return loops
LOOPS=_fill_holes(CV,TRI)


def euler(R):
    yaw=np.degrees(np.arcsin(np.clip(R[0,2],-1,1))); pitch=np.degrees(np.arctan2(-R[1,2],R[2,2])); roll=np.degrees(np.arctan2(-R[0,1],R[0,0]))
    return dict(yaw=float(yaw),pitch=float(pitch),roll=float(roll))

def reconstruct(img: Image.Image) -> dict:
    rgb=np.array(img.convert('RGB')); h,w=rgb.shape[:2]
    _,meta=extract_face_bbox_and_crop(img.convert('RGB'))
    P,score=landmarks468(rgb,meta['bbox'])
    obs=np.c_[P[:,0],-P[:,1],-P[:,2]].astype(np.float64)           # x right, y up, z toward camera
    R,s,t=umeyama(CAN468.astype(np.float64),obs)
    G=((obs-t)@R)/s                                                  # person-specific shape, pose-normalised (frontal)
    # light Laplacian smoothing of depth only (keeps x,y landmark-exact)
    z=G[:,2].copy()
    for _ in range(2):
        z=np.array([ (1-0.35)*z[i]+0.35*z[ADJ[i][ADJ[i]<468]].mean() if len(ADJ[i][ADJ[i]<468]) else z[i] for i in range(468)])
    G[:,2]=z
    V=CV.astype(np.float64).copy(); V[:468]=G
    # propagate face deformation into cranial vault (kNN gaussian blend of residuals)
    res=G-CAN468; tree_pts=CAN468
    for i in np.where(CRAN)[0]:
        d=np.linalg.norm(tree_pts-CV[i],axis=1); idx=np.argsort(d)[:10]; wgt=np.exp(-(d[idx]/0.08)**2)+1e-6
        V[i]=CV[i]+(res[idx]*wgt[:,None]).sum(0)/wgt.sum()*np.exp(-(d[idx].min()/0.25)**2)
    uv=np.zeros((len(CV),2),np.float32); uv[:468]=P[:,:2]
    for i in np.where(FILL)[0]:
        nb=ADJ[i][ADJ[i]<468]; uv[i]=P[nb,:2].mean(0)
    # cranial base colour: skin/hair strip just above the forehead
    fx,fy=P[10,0],P[10,1]; bh=meta['bbox'][3]
    y0=int(np.clip(fy-0.12*bh,0,h-1)); y1=int(np.clip(fy-0.03*bh,y0+1,h)); x0=int(np.clip(fx-0.1*bh,0,w-1)); x1=int(np.clip(fx+0.1*bh,x0+1,w))
    cran_col=np.median(rgb[y0:y1,x0:x1].reshape(-1,3),0)
    # feather weight: 0 on face-oval ring (adjacent to skull), rising to 1 three rings inside
    ring=np.full(len(CV),99.); frontier=[i for i in range(468) if (BACK[ADJ[i]]).any()]
    for i in frontier: ring[i]=0
    q=list(frontier)
    while q:
        nq=[]
        for i in q:
            for j in ADJ[i]:
                if j<468 and ring[j]>ring[i]+1: ring[j]=ring[i]+1; nq.append(j)
        q=nq
    wt=np.clip((ring-0.0)/4.0,0.0,1.0); wt[468:]=1.0
    # skull colour: sampled hair/forehead tone, darker toward the back for depth
    zb=np.clip((V[:,2]+0.48)/0.5,0,1); bcol=(cran_col[None,:]*(0.55+0.45*zb[:,None])).astype(np.float32)
    return dict(V=V,uv=uv,rgb=rgb,P=P,pose=euler(R),scale=s,score=score,meta=meta,cran_col=cran_col,R=R,s=s,t=t,wt=wt,bcol=bcol)

def normals(V,tri):
    n=np.zeros_like(V); a,b,c=V[tri[:,0]],V[tri[:,1]],V[tri[:,2]]; fn=np.cross(b-a,c-a)
    for k in range(3): np.add.at(n,tri[:,k],fn)
    return n/np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)

def rot(yaw=0,pitch=0,roll=0):
    y,p,r=np.radians([yaw,pitch,roll])
    Ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    Rx=np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
    Rz=np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])
    return Rz@Ry@Rx

def render(V,uv,rgb,cran_col,yaw=0,pitch=0,size=512,vert_color=None,tri=TRI,shade=True,bg=(22,24,30),wt=None,bcol=None,back=BACK):
    """Orthographic z-buffer render. Returns image and projected vertex xy (px)."""
    Rv=rot(yaw,pitch); Vr=V@Rv.T; N=normals(Vr,tri)
    sc=size*0.42/np.abs(Vr[:,:2]).max(); X=Vr[:,0]*sc+size/2; Y=-Vr[:,1]*sc+size/2+size*0.03
    zb=np.full((size,size),-1e9); uvb=np.full((size,size,2),-1.,np.float32); nb=np.zeros((size,size,3),np.float32)
    wbuf=np.ones((size,size),np.float32); cb=np.zeros((size,size,3),np.float32); hascol=np.zeros((size,size),bool); m=np.zeros((size,size),bool)
    for f in tri:
        x,y,z=X[f],Y[f],Vr[f,2]
        x0,x1=int(max(0,np.floor(x.min()))),int(min(size-1,np.ceil(x.max()))); y0,y1=int(max(0,np.floor(y.min()))),int(min(size-1,np.ceil(y.max())))
        if x1<x0 or y1<y0: continue
        den=(y[1]-y[2])*(x[0]-x[2])+(x[2]-x[1])*(y[0]-y[2])
        if abs(den)<1e-9: continue
        gx,gy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
        l0=((y[1]-y[2])*(gx-x[2])+(x[2]-x[1])*(gy-y[2]))/den; l1=((y[2]-y[0])*(gx-x[2])+(x[0]-x[2])*(gy-y[2]))/den; l2=1-l0-l1
        ins=(l0>=-1e-4)&(l1>=-1e-4)&(l2>=-1e-4)
        if not ins.any(): continue
        zz=l0*z[0]+l1*z[1]+l2*z[2]; sl=(slice(y0,y1+1),slice(x0,x1+1))
        upd=ins&(zz>zb[sl]); 
        if not upd.any(): continue
        zb[sl][upd]=zz[upd]; m[sl][upd]=True
        nn=l0[...,None]*N[f[0]]+l1[...,None]*N[f[1]]+l2[...,None]*N[f[2]]; nb[sl][upd]=nn[upd]
        if vert_color is not None:
            cc=l0[...,None]*vert_color[f[0]]+l1[...,None]*vert_color[f[1]]+l2[...,None]*vert_color[f[2]]; cb[sl][upd]=cc[upd]; hascol[sl][upd]=True
        elif back[f].any():
            bc=bcol if bcol is not None else np.tile(np.asarray(cran_col,np.float32),(len(V),1))
            cc=l0[...,None]*bc[f[0]]+l1[...,None]*bc[f[1]]+l2[...,None]*bc[f[2]]; cb[sl][upd]=cc[upd]; hascol[sl][upd]=True
            wbuf[sl][upd]=0.0
        else:
            u=l0[...,None]*uv[f[0]]+l1[...,None]*uv[f[1]]+l2[...,None]*uv[f[2]]; uvb[sl][upd]=u[upd]; hascol[sl][upd]=False
            ww=1.0 if wt is None else (l0*wt[f[0]]+l1*wt[f[1]]+l2*wt[f[2]]); wbuf[sl][upd]=(ww if np.isscalar(ww) else ww[upd])
            if bcol is not None:
                cc=l0[...,None]*bcol[f[0]]+l1[...,None]*bcol[f[1]]+l2[...,None]*bcol[f[2]]; cb[sl][upd]=cc[upd]
    out=np.zeros((size,size,3),np.float32); out[:]=bg
    tex=cv2.remap(rgb,uvb[...,0].astype(np.float32),uvb[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    col=np.where(hascol[...,None],cb,tex*wbuf[...,None]+cb*(1-wbuf[...,None]))
    if shade:
        nb/=np.maximum(np.linalg.norm(nb,axis=2,keepdims=True),1e-9)
        Ldir=np.array([-0.35,0.45,0.82]);Ldir/=np.linalg.norm(Ldir)
        lam=np.clip(nb@Ldir,0,1); view=np.array([0,0,1.]); hv=(Ldir+view);hv/=np.linalg.norm(hv)
        spec=np.clip(nb@hv,0,1)**40*0.12
        light=0.62+0.38*lam
        col=np.clip(col*light[...,None]+255*spec[...,None],0,255)
    out[m]=col[m]
    return out.astype(np.uint8),np.c_[X,Y],m


def export_obj(r: dict, path: str, tri=None):
    """Write the reconstructed (pose-normalised) head as a textured Wavefront .obj (UV = photo px / image size)."""
    tri=TRI if tri is None else tri; V=r["V"]; h,w=r["rgb"].shape[:2]; N=normals(V,tri)
    L=["# GECO MediaPipe-468 3D face"]+[f"v {a:.5f} {b:.5f} {c:.5f}" for a,b,c in V]
    L+=[f"vt {u/w:.5f} {1-v/h:.5f}" for u,v in r["uv"]]+[f"vn {a:.4f} {b:.4f} {c:.4f}" for a,b,c in N]
    L+=[f"f {a+1}/{a+1}/{a+1} {b+1}/{b+1}/{b+1} {c+1}/{c+1}/{c+1}" for a,b,c in tri]
    Path(path).write_text("\n".join(L))

def residual_map(r: dict, yaw: float = 0.0, size: int = 512) -> np.ndarray:
    """Per-vertex anomaly: re-detect the 468 landmarks on a render of the mesh and measure the error vs the mesh's
    own projection, in % of inter-ocular distance. High = region where geometry/texture disagrees with a real face."""
    im,xy,_=render(r["V"],r["uv"],r["rgb"],r["cran_col"],yaw,0,size=size,wt=r["wt"],bcol=r["bcol"])
    _,mt=extract_face_bbox_and_crop(Image.fromarray(im))
    if not mt or not mt.get("is_detected"): return np.full(468,np.nan)
    Q,_=landmarks468(im,mt["bbox"]); iod=np.linalg.norm(xy[33]-xy[263])
    return 100*np.linalg.norm(Q[:468,:2]-xy[:468],axis=1)/iod
