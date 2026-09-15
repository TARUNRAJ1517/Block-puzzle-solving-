import cv2
import numpy as np
from PIL import Image

def read_rgb(uploaded):
    return np.array(Image.open(uploaded).convert("RGB"))

def order_quad(pts):
    pts=np.asarray(pts,dtype=np.float32)
    s=pts.sum(axis=1)
    d=np.diff(pts,axis=1).ravel()
    return np.array([pts[np.argmin(s)],pts[np.argmin(d)],
                     pts[np.argmax(s)],pts[np.argmax(d)]],np.float32)

def find_board(img):
    bgr=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    mask=(((hsv[:,:,0]<12)|(hsv[:,:,0]>170))&(hsv[:,:,1]>70)&(hsv[:,:,2]<190)).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((21,21),np.uint8))
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    H,W=mask.shape
    cand=[]
    for c in cnts:
        area=cv2.contourArea(c)
        if area < H*W*.03: continue
        peri=cv2.arcLength(c,True)
        q=cv2.approxPolyDP(c,.035*peri,True)
        if len(q)!=4: continue
        x,y,w,h=cv2.boundingRect(q)
        ratio=w/max(h,1)
        if .75<ratio<1.35:
            cand.append((area,q.reshape(4,2)))
    return order_quad(max(cand,key=lambda z:z[0])[1]) if cand else None

def warp_board(img,quad,size=800):
    dst=np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]],np.float32)
    M=cv2.getPerspectiveTransform(quad,dst)
    return cv2.warpPerspective(img,M,(size,size))

def board_from_warp(warp):
    b=np.zeros((8,8),np.uint8)
    s=warp.shape[0]//8
    for r in range(8):
        for c in range(8):
            crop=warp[int((r+.28)*s):int((r+.72)*s),
                      int((c+.28)*s):int((c+.72)*s)]
            hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
            sat=float(np.median(hsv[:,:,1]))
            val=float(np.median(hsv[:,:,2]))
            b[r,c]=1 if sat>105 and val>90 else 0
    return b

def detect_piece_components(img, board_quad):
    H,W=img.shape[:2]
    y0=int(np.max(board_quad[:,1]))+15 if board_quad is not None else int(H*.55)
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    mask=((hsv[:,:,1]>115)&(hsv[:,:,2]>100)).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,lab,stats,cent=cv2.connectedComponentsWithStats(mask,8)
    comps=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if y>=y0 and area>80 and w>10 and h>10:
            comps.append((x,y,w,h,area))
    return sorted(comps,key=lambda z:z[0])

def component_to_piece(img, comp):
    x,y,w,h,_=comp
    crop=img[max(0,y-3):y+h+3,max(0,x-3):x+w+3]
    hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
    mask=((hsv[:,:,1]>115)&(hsv[:,:,2]>100)).astype(np.uint8)
    # Find block centers from connected components inside the piece.
    n,lab,stats,cent=cv2.connectedComponentsWithStats(mask,8)
    centers=[]
    for i in range(1,n):
        xx,yy,ww,hh,area=stats[i]
        if area>40:
            centers.append((xx+ww/2,yy+hh/2))
    if not centers:
        return None
    # Cluster centers onto a regular block grid using the median block pitch.
    xs=sorted(p[0] for p in centers); ys=sorted(p[1] for p in centers)
    diffs=[abs(xs[i]-xs[i-1]) for i in range(1,len(xs)) if abs(xs[i]-xs[i-1])>5]
    diffs += [abs(ys[i]-ys[i-1]) for i in range(1,len(ys)) if abs(ys[i]-ys[i-1])>5]
    pitch=float(np.median(diffs)) if diffs else max(15,min(w,h)/2)
    xbase=min(xs); ybase=min(ys)
    cells=[]
    for cx,cy in centers:
        cells.append((int(round((cy-ybase)/pitch)),
                      int(round((cx-xbase)/pitch))))
    cells=tuple(sorted(set(cells)))
    return cells if cells else None
