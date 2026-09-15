from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Piece:
    cells: tuple
    name: str = "piece"

def normalize(cells):
    if not cells:
        return ()
    mr = min(r for r,c in cells)
    mc = min(c for r,c in cells)
    return tuple(sorted((r-mr,c-mc) for r,c in cells))

def placements(board, piece):
    H, W = board.shape
    p = normalize(piece.cells)
    ph = max(r for r,c in p) + 1
    pw = max(c for r,c in p) + 1
    out = []
    for r0 in range(H-ph+1):
        for c0 in range(W-pw+1):
            coords = tuple((r0+r,c0+c) for r,c in p)
            if all(board[r,c] == 0 for r,c in coords):
                nb = board.copy()
                for r,c in coords:
                    nb[r,c] = 1
                rows = np.where(nb.all(axis=1))[0].tolist()
                cols = np.where(nb.all(axis=0))[0].tolist()
                if rows:
                    nb[rows,:] = 0
                if cols:
                    nb[:,cols] = 0
                out.append((nb, coords, rows, cols))
    return out

def board_score(b):
    filled = int(b.sum())
    heights = []
    for c in range(b.shape[1]):
        ys = np.where(b[:,c])[0]
        heights.append(0 if len(ys)==0 else b.shape[0]-ys[0])
    rough = sum(abs(heights[i]-heights[i+1]) for i in range(7))
    holes = 0
    for c in range(8):
        ys = np.where(b[:,c])[0]
        if len(ys):
            holes += int((~b[ys[0]:,c]).sum())
    return -1.5*filled - .35*rough - 1.25*holes

def solve(board, pieces):
    best = None

    def dfs(b, remaining, path, total):
        nonlocal best
        if not remaining:
            score = total + board_score(b)
            if best is None or score > best["score"]:
                best = {"score":score, "board":b, "moves":path}
            return
        for pi in remaining:
            for nb, coords, rows, cols in placements(b, pieces[pi]):
                bonus = 100*(len(rows)+len(cols)) + 25*(len(rows)*len(cols))
                move = {"piece_index":pi, "coords":coords,
                        "cleared_rows":rows, "cleared_cols":cols}
                dfs(nb, tuple(x for x in remaining if x != pi),
                    path+[move], total+bonus+board_score(nb))
    dfs(board.astype(np.uint8), tuple(range(len(pieces))), [], 0)
    return best
