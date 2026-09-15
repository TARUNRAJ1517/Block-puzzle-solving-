# Block Puzzle Screenshot Solver

Upload a screenshot of the supported 8x8 Block Blast-style game. The app detects the board and the three offered pieces, then exhaustively searches all piece orders and legal placements.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Deploy on Render
Push this folder to GitHub, then create a new Render Web Service from the repo. Select Docker. Render can also use `render.yaml`.

The app deliberately shows no move when it cannot verify exactly three pieces or the board.
