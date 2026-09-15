import streamlit as st
import numpy as np
import cv2
from vision import read_rgb,find_board,warp_board,board_from_warp,detect_piece_components,component_to_piece
from solver import Piece,solve

st.set_page_config(page_title="Block Puzzle Solver",layout="wide")
st.title("🧩 Screenshot Block Puzzle Solver")
st.write("Upload a current screenshot. The app must verify the board and all 3 pieces before giving a move.")

up=st.file_uploader("Upload screenshot",type=["png","jpg","jpeg","webp"])
if not up:
    st.stop()

img=read_rgb(up)
quad=find_board(img)
if quad is None:
    st.error("Board not detected reliably — no move will be guessed.")
    st.image(img,use_container_width=True)
    st.stop()

warp=warp_board(img,quad)
board=board_from_warp(warp)
comps=detect_piece_components(img,quad)
pieces=[]
for comp in comps:
    cells=component_to_piece(img,comp)
    if cells:
        pieces.append(Piece(cells,f"piece {len(pieces)+1}"))

# Keep the three largest/left-to-right offered pieces.
pieces=pieces[:3]

st.subheader("Detected 8×8 board")
st.code("\n".join("".join("■" if x else "·" for x in row) for row in board))
st.write("Verified offered pieces:",len(pieces))

if len(pieces)!=3:
    st.warning("All 3 pieces were not verified. No answer is shown.")
    st.image(img,use_container_width=True)
    st.stop()

result=solve(board,pieces)
if not result:
    st.error("No legal sequence for all three detected pieces.")
    st.stop()

st.success("Verified 3 pieces. Best legal 3-piece sequence found.")
for n,m in enumerate(result["moves"],1):
    st.write(f"**Move {n}: Piece {m['piece_index']+1}** — "
             f"place at {m['coords']} | clears rows {m['cleared_rows']} | columns {m['cleared_cols']}")

vis=warp.copy()
S=vis.shape[0]//8
for n,m in enumerate(result["moves"],1):
    for r,c in m["coords"]:
        cv2.rectangle(vis,(c*S+8,r*S+8),((c+1)*S-8,(r+1)*S-8),(255,255,255),5)
        cv2.putText(vis,str(n),(c*S+S//3,r*S+2*S//3),
                    cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,0,0),5,cv2.LINE_AA)
st.image(vis,caption="1 → 2 → 3",use_container_width=True)
