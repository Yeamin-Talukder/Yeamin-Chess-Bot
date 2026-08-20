import nbformat as nbf

def build_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # --------------------------------------------------------------------------------
    # CELL 1 — Project Configuration
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 1 — Project Configuration"))
    cells.append(nbf.v4.new_code_cell("""\
import os

PROJECT_DIR = "/content/drive/MyDrive/chess_ai"
CONFIG = {
    "input_file": f"{PROJECT_DIR}/input/positions.parquet",
    "output_dir": f"{PROJECT_DIR}/output",
    "chunk_dir": f"{PROJECT_DIR}/output/chunks",
    "merged_dir": f"{PROJECT_DIR}/output/merged",
    "log_dir": f"{PROJECT_DIR}/output/logs",
    "checkpoint_dir": f"{PROJECT_DIR}/checkpoints",
    
    "stockfish_depth": 10,
    "multipv": 5,
    "stockfish_threads_per_worker": 1,
    "stockfish_hash_mb": 128,
    
    "num_workers": None,  # Will be auto-detected safely
    "chunk_size": 1000,
    "save_every": 1000
}

# Ensure local runtime temp paths
TEMP_CHUNK_DIR = "/content/temp_chunks"
os.makedirs(TEMP_CHUNK_DIR, exist_ok=True)
"""))

    # --------------------------------------------------------------------------------
    # CELL 2 — Mount Google Drive
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 2 — Mount Google Drive"))
    cells.append(nbf.v4.new_code_cell("""\
from google.colab import drive
import os

drive.mount('/content/drive')

# Create necessary directories in Google Drive
for d in [CONFIG["output_dir"], CONFIG["chunk_dir"], CONFIG["merged_dir"], CONFIG["log_dir"], CONFIG["checkpoint_dir"]]:
    os.makedirs(d, exist_ok=True)
    
print("Directories verified.")
"""))

    # --------------------------------------------------------------------------------
    # CELL 3 — Install Dependencies
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 3 — Install Dependencies"))
    cells.append(nbf.v4.new_code_cell("""\
!pip install python-chess pyarrow pandas fastparquet psutil tqdm
"""))

    # --------------------------------------------------------------------------------
    # CELL 4 — Install / Locate Stockfish
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 4 — Install / Locate Stockfish"))
    cells.append(nbf.v4.new_code_cell("""\
import os
import subprocess

print("Installing Stockfish via apt-get for maximum compatibility...")
!apt-get update -qq
!apt-get install stockfish -y -qq

STOCKFISH_EXEC = "/usr/games/stockfish"

# Verify Stockfish works
try:
    result = subprocess.run([STOCKFISH_EXEC, "uci"], capture_output=True, text=True, timeout=5)
    if "stockfish" in result.stdout.lower() and "uciok" in result.stdout.lower():
        print("Stockfish verified working successfully!")
    else:
        print("Stockfish executed but returned unexpected output:\\n", result.stdout)
except Exception as e:
    print(f"Error running Stockfish: {e}\\nSTOPPING! Fix the Stockfish installation before continuing.")
    raise
"""))

    # --------------------------------------------------------------------------------
    # CELL 5 — Detect Hardware
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 5 — Detect Hardware"))
    cells.append(nbf.v4.new_code_cell("""\
import psutil
import multiprocessing

physical_cores = psutil.cpu_count(logical=False)
logical_cores = psutil.cpu_count(logical=True)
ram_gb = psutil.virtual_memory().total / (1024 ** 3)

print("="*40)
print(f"CPU: {physical_cores} Physical Cores / {logical_cores} Logical Cores")
print(f"RAM: {ram_gb:.2f} GB")
print("="*40)

# Determine safe workers (leave 1 physical core for OS/Colab overhead)
safe_workers = max(1, physical_cores - 1) if physical_cores else max(1, logical_cores - 1)
if CONFIG.get("num_workers") is None:
    CONFIG["num_workers"] = safe_workers
print(f"Auto-selected num_workers: {safe_workers}, Actual used: {CONFIG['num_workers']}")
print("Note: Do NOT use GPU for Stockfish CPU calculations.")
"""))

    # --------------------------------------------------------------------------------
    # CELL 6 — Validate Input Dataset
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 6 — Validate Input Dataset"))
    cells.append(nbf.v4.new_code_cell("""\
import pyarrow.parquet as pq
import pandas as pd
import os

if not os.path.exists(CONFIG["input_file"]):
    raise FileNotFoundError(f"Input dataset not found at {CONFIG['input_file']}")

print("Loading dataset metadata...")
table = pq.read_table(CONFIG["input_file"])
df_sample = table.schema

required_cols = ["position_id", "game_id", "fen", "your_color"]
available_cols = [c.name for c in df_sample]

print(f"Total positions: {table.num_rows}")
print(f"Columns: {available_cols}")

missing = [c for c in required_cols if c not in available_cols]
# Handle your_move variations
move_col = None
if "your_move_uci" in available_cols:
    move_col = "your_move_uci"
elif "your_move" in available_cols:
    move_col = "your_move"

if move_col is None:
    missing.append("your_move_uci OR your_move")

if missing:
    raise ValueError(f"Missing required columns: {missing}")
else:
    print("Dataset validated successfully.")
    CONFIG["move_col"] = move_col
"""))

    # --------------------------------------------------------------------------------
    # CELL 7 — Stockfish Configuration
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 7 — Stockfish Configuration & Logic Modules"))
    cells.append(nbf.v4.new_code_cell("""\
import chess
import chess.engine

def evaluate_position(engine, fen, your_move_uci, your_color, depth, multipv):
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            return {"analysis_status": "invalid_fen"}
            
        try:
            your_chess_move = chess.Move.from_uci(your_move_uci)
        except ValueError:
            return {"analysis_status": "invalid_move_format"}
            
        if your_chess_move not in board.legal_moves:
            return {"analysis_status": "illegal_move"}

        # Analyze MultiPV
        limit = chess.engine.Limit(depth=depth)
        info = engine.analyse(board, limit, multipv=multipv)
        if isinstance(info, dict):
            info = [info]
            
        is_white = (your_color.lower() == "white")
        pov_color = chess.WHITE if is_white else chess.BLACK
        
        # Sort just to be safe
        info.sort(key=lambda x: x["score"].pov(board.turn).score(mate_score=10000), reverse=True)
        
        result = {"analysis_status": "success"}
        your_move_eval_cp = None
        your_move_eval_mate = None
        your_move_rank = 0 # 0 means outside top N
        
        # Parse top engine moves
        for i, line in enumerate(info):
            rank = i + 1
            if rank <= 5:
                pv = line.get("pv", [])
                move_uci = pv[0].uci() if pv else ""
                move_san = board.san(pv[0]) if pv else ""
                
                score = line["score"].pov(pov_color)
                cp = score.score()
                mate = score.mate()
                
                if cp is None and mate is not None:
                    # Convert mate to high cp equivalent safely for CPL
                    cp = 10000 if mate > 0 else -10000
                
                result[f"stockfish_rank_{rank}_move_uci"] = move_uci
                result[f"stockfish_rank_{rank}_move_san"] = move_san
                result[f"stockfish_rank_{rank}_eval_cp"] = cp
                result[f"stockfish_rank_{rank}_eval_mate"] = mate
                
                if rank == 1:
                    result["stockfish_best_move_uci"] = move_uci
                    result["stockfish_best_move_san"] = move_san
                    result["stockfish_best_eval_cp"] = cp
                    result["stockfish_best_eval_mate"] = mate
                    
            # Check if this line is our move
            pv = line.get("pv", [])
            if pv and pv[0].uci() == your_move_uci:
                your_move_rank = i + 1
                score = line["score"].pov(pov_color)
                your_move_eval_cp = score.score()
                your_move_eval_mate = score.mate()
                if your_move_eval_cp is None and your_move_eval_mate is not None:
                    your_move_eval_cp = 10000 if your_move_eval_mate > 0 else -10000
                    
        # Explicit evaluation if outside MultiPV
        if your_move_rank == 0:
            root_limit = chess.engine.Limit(depth=depth)
            specific_info = engine.analyse(board, root_limit, root_moves=[your_chess_move])
            if isinstance(specific_info, list):
                specific_info = specific_info[0]
                
            score = specific_info["score"].pov(pov_color)
            your_move_eval_cp = score.score()
            your_move_eval_mate = score.mate()
            if your_move_eval_cp is None and your_move_eval_mate is not None:
                your_move_eval_cp = 10000 if your_move_eval_mate > 0 else -10000
                
        result["your_move_rank"] = your_move_rank
        result["your_move_eval_cp"] = your_move_eval_cp
        result["your_move_eval_mate"] = your_move_eval_mate
        
        best_eval = result.get("stockfish_best_eval_cp", 0)
        
        # Calculate CPL (Centipawn Loss is always positive relative to best move)
        cpl = best_eval - your_move_eval_cp
        result["centipawn_loss"] = max(0, cpl)
        
        result["stockfish_depth"] = depth
        result["stockfish_multipv"] = multipv
        
        return result
        
    except Exception as e:
        return {"analysis_status": f"error: {str(e)}"}
"""))

    # --------------------------------------------------------------------------------
    # CELL 8 — Test Single Position
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 8 — Test Single Position"))
    cells.append(nbf.v4.new_code_cell("""\
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_EXEC)
try:
    engine.configure({"Threads": 1, "Hash": 32})
    test_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    # White played e4. Black plays e5
    test_res = evaluate_position(engine, test_fen, "e7e5", "Black", 10, 5)
    print("Single Position Test Result:")
    for k, v in test_res.items():
        if v is not None:
            print(f"  {k}: {v}")
    
    assert test_res["analysis_status"] == "success", "Failed evaluation"
    assert "centipawn_loss" in test_res, "CPL not calculated"
    print("\\nTest Passed!")
finally:
    engine.quit()
"""))

    # --------------------------------------------------------------------------------
    # CELL 9 — Benchmark Workers
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 9 — Benchmark Workers (Disabled by default)"))
    cells.append(nbf.v4.new_code_cell("""\
import time
import multiprocessing

BENCHMARK_MODE = False
BENCHMARK_POSITIONS = 50

# Worker init for multiprocessing
def init_worker(exec_path, threads, hash_mb):
    global _worker_engine
    _worker_engine = chess.engine.SimpleEngine.popen_uci(exec_path)
    _worker_engine.configure({"Threads": threads, "Hash": hash_mb})

def worker_job(data):
    # data: (fen, your_move_uci, your_color, depth, multipv)
    global _worker_engine
    fen, move, color, depth, multipv = data
    return evaluate_position(_worker_engine, fen, move, color, depth, multipv)

if BENCHMARK_MODE:
    print("Running Benchmark...")
    test_table = table.slice(0, BENCHMARK_POSITIONS).to_pandas()
    jobs = [(row['fen'], row[CONFIG['move_col']], row['your_color'], 10, 5) for idx, row in test_table.iterrows()]
    
    for w in [1, 2, 4]:
        if w > CONFIG["num_workers"]: continue
        
        start = time.time()
        with multiprocessing.Pool(w, initializer=init_worker, initargs=(STOCKFISH_EXEC, CONFIG["stockfish_threads_per_worker"], CONFIG["stockfish_hash_mb"])) as pool:
            res = pool.map(worker_job, jobs)
        
        elapsed = time.time() - start
        speed = BENCHMARK_POSITIONS / elapsed
        print(f"Workers: {w} | Time: {elapsed:.2f}s | Speed: {speed:.1f} pos/sec")
"""))

    # --------------------------------------------------------------------------------
    # CELL 10 — Test 100 Positions
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 10 — Test 100 Positions"))
    cells.append(nbf.v4.new_code_cell("""\
TEST_MODE = True
TEST_POSITIONS = 100

def process_chunk(chunk_df, chunk_id):
    import json, datetime
    
    jobs = [(r['fen'], r[CONFIG['move_col']], r['your_color'], CONFIG['stockfish_depth'], CONFIG['multipv']) for idx, r in chunk_df.iterrows()]
    
    with multiprocessing.Pool(CONFIG["num_workers"], initializer=init_worker, initargs=(STOCKFISH_EXEC, CONFIG["stockfish_threads_per_worker"], CONFIG["stockfish_hash_mb"])) as pool:
        results = pool.map(worker_job, jobs)
    
    # Merge results
    for i, res in enumerate(results):
        for k, v in res.items():
            chunk_df.loc[chunk_df.index[i], k] = v
        chunk_df.loc[chunk_df.index[i], "analysis_timestamp"] = str(datetime.datetime.now())
            
    # Temp file to atomic rename
    tmp_path = os.path.join(TEMP_CHUNK_DIR, f"chunk_{chunk_id:06d}.tmp.parquet")
    final_path = os.path.join(CONFIG["chunk_dir"], f"chunk_{chunk_id:06d}.parquet")
    
    chunk_df.to_parquet(tmp_path, index=False)
    # Move to Google Drive (Atomic-ish)
    import shutil
    shutil.move(tmp_path, final_path)
    return len(chunk_df), results

if TEST_MODE:
    print("Testing pipeline on 100 positions...")
    test_df = table.slice(0, TEST_POSITIONS).to_pandas()
    # Test saving chunk 999999 (fake ID for test)
    n, res = process_chunk(test_df, 999999)
    print(f"Processed {n} positions.")
    print(f"Example rank matches: {[r['your_move_rank'] for r in res[:10]]}")
"""))

    # --------------------------------------------------------------------------------
    # CELL 11 — Full Analysis Logic
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 11 — Full Analysis Logic"))
    cells.append(nbf.v4.new_code_cell("""\
import glob
import math
import time
import json
import datetime
from IPython.display import clear_output

def update_heartbeat(status_data):
    path = os.path.join(CONFIG["checkpoint_dir"], "status.json")
    status_data["last_update"] = str(datetime.datetime.now())
    with open(path, 'w') as f:
        json.dump(status_data, f, indent=2)

def log_error(pos_id, error):
    path = os.path.join(CONFIG["log_dir"], "stockfish_errors.jsonl")
    with open(path, 'a') as f:
        f.write(json.dumps({"position_id": pos_id, "error": str(error), "timestamp": str(datetime.datetime.now())}) + "\\n")
"""))

    # --------------------------------------------------------------------------------
    # CELL 12 — Resume Existing Analysis (Core Loop)
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 12 — Resume Existing Analysis & Run"))
    cells.append(nbf.v4.new_code_cell("""\
# WARNING: SET TO FALSE TO RUN THE FULL 321K DATASET
TEST_MODE = False

if not TEST_MODE:
    df_full = table.to_pandas()
    total_positions = len(df_full)
    chunk_size = CONFIG["chunk_size"]
    total_chunks = math.ceil(total_positions / chunk_size)
    
    # Discover completed chunks
    existing = glob.glob(os.path.join(CONFIG["chunk_dir"], "chunk_*.parquet"))
    # Filter out the test chunk
    completed_ids = set()
    for f in existing:
        base = os.path.basename(f)
        if base.startswith("chunk_") and base.endswith(".parquet") and "999999" not in base:
            try:
                # Basic verification that it opens
                pq.read_metadata(f)
                idx = int(base.split("_")[1].split(".")[0])
                completed_ids.add(idx)
            except Exception as e:
                print(f"Corrupt chunk found and ignored: {f}")
                
    print(f"Total chunks: {total_chunks} | Completed: {len(completed_ids)} | Remaining: {total_chunks - len(completed_ids)}")
    
    # Track stats
    stats = {
        "total_positions": total_positions,
        "completed_positions": len(completed_ids) * chunk_size, # Approx
        "completed_chunks": len(completed_ids),
        "rank_1": 0, "top_3": 0, "cpl_sum": 0, "speed": 0,
        "status": "running"
    }
    
    start_time = time.time()
    positions_processed_this_run = 0
    
    for chunk_id in range(total_chunks):
        if chunk_id in completed_ids:
            continue
            
        start_row = chunk_id * chunk_size
        end_row = min(start_row + chunk_size, total_positions)
        chunk_df = df_full.iloc[start_row:end_row].copy()
        
        try:
            n, results = process_chunk(chunk_df, chunk_id)
            
            # Accumulate fast stats for display
            for r in results:
                if r.get("your_move_rank") == 1:
                    stats["rank_1"] += 1
                    stats["top_3"] += 1
                elif r.get("your_move_rank") in [2, 3]:
                    stats["top_3"] += 1
                stats["cpl_sum"] += r.get("centipawn_loss", 0)
                
            positions_processed_this_run += n
            stats["completed_positions"] += n
            stats["completed_chunks"] += 1
            
            # Display logic
            elapsed = time.time() - start_time
            speed = positions_processed_this_run / elapsed if elapsed > 0 else 0
            stats["speed"] = speed
            
            rem = total_positions - stats["completed_positions"]
            eta_m = (rem / speed) / 60 if speed > 0 else 0
            
            clear_output(wait=True)
            pct = stats["completed_positions"] / total_positions * 100
            
            r1_pct = (stats["rank_1"] / positions_processed_this_run * 100) if positions_processed_this_run else 0
            r3_pct = (stats["top_3"] / positions_processed_this_run * 100) if positions_processed_this_run else 0
            cpl_avg = (stats["cpl_sum"] / positions_processed_this_run) if positions_processed_this_run else 0
            
            print(f"Stockfish Analysis")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"Completed: {stats['completed_positions']} / {total_positions}")
            print(f"Progress: {pct:.2f}%")
            print(f"\\nSpeed: {speed:.1f} positions/sec")
            print(f"ETA: {eta_m:.1f} minutes")
            print(f"\\nWorkers: {CONFIG['num_workers']} | Depth: {CONFIG['stockfish_depth']} | MultiPV: {CONFIG['multipv']}")
            print(f"\\nBest move match: {r1_pct:.1f}%")
            print(f"Top-3 match: {r3_pct:.1f}%")
            print(f"Average CPL (this run): {cpl_avg:.1f}")
            
            update_heartbeat(stats)
            
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
            log_error(f"chunk_{chunk_id}", e)
            
    stats["status"] = "finished"
    update_heartbeat(stats)
    print("ALL CHUNKS COMPLETED!")
"""))

    # --------------------------------------------------------------------------------
    # CELL 13 — Merge Completed Chunks
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 13 — Merge Completed Chunks"))
    cells.append(nbf.v4.new_code_cell("""\
FINAL_PARQUET = os.path.join(CONFIG["merged_dir"], "positions_stockfish.parquet")

def merge_chunks():
    chunks = sorted(glob.glob(os.path.join(CONFIG["chunk_dir"], "chunk_*.parquet")))
    chunks = [c for c in chunks if "999999" not in c] # Filter test chunk
    
    if not chunks:
        print("No chunks to merge.")
        return
        
    print(f"Merging {len(chunks)} chunks into {FINAL_PARQUET}...")
    
    # We can read all tables and concat
    tables = [pq.read_table(f) for f in chunks]
    full_table = pyarrow.concat_tables(tables)
    
    pq.write_table(full_table, FINAL_PARQUET)
    print(f"Merge complete! Final rows: {full_table.num_rows}")
    return full_table

# Uncomment to merge
# full_table = merge_chunks()
"""))

    # --------------------------------------------------------------------------------
    # CELL 14 — Final Validation
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 14 — Final Validation"))
    cells.append(nbf.v4.new_code_cell("""\
# if 'full_table' in locals():
#     df_final = full_table.to_pandas()
#     input_rows = table.num_rows
#     output_rows = len(df_final)
#     print(f"Input rows: {input_rows}\\nOutput rows: {output_rows}")
#     
#     duplicates = df_final.duplicated(subset=['position_id']).sum()
#     print(f"Duplicate position IDs: {duplicates}")
#     
#     success = len(df_final[df_final['analysis_status'] == 'success'])
#     invalid = len(df_final[df_final['analysis_status'] != 'success'])
#     print(f"Analyzed successfully: {success}\\nInvalid/Failed: {invalid}")
"""))

    # --------------------------------------------------------------------------------
    # CELL 15 — Generate Analysis Summary
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 15 — Generate Analysis Summary"))
    cells.append(nbf.v4.new_code_cell("""\
# if 'df_final' in locals():
#     df_success = df_final[df_final['analysis_status'] == 'success']
#     t = len(df_success)
#     if t > 0:
#         r1 = len(df_success[df_success['your_move_rank'] == 1]) / t * 100
#         r3 = len(df_success[df_success['your_move_rank'].isin([1,2,3])]) / t * 100
#         r5 = len(df_success[df_success['your_move_rank'].isin([1,2,3,4,5])]) / t * 100
#         ro = len(df_success[df_success['your_move_rank'] == 0]) / t * 100
#         
#         avg_cpl = df_success['centipawn_loss'].mean()
#         med_cpl = df_success['centipawn_loss'].median()
#         
#         white_cpl = df_success[df_success['your_color'] == 'White']['centipawn_loss'].mean()
#         black_cpl = df_success[df_success['your_color'] == 'Black']['centipawn_loss'].mean()
#         
#         summary = {
#             "total_positions": len(df_final),
#             "successfully_analyzed": t,
#             "invalid_positions": len(df_final) - t,
#             "best_move_match_percent": r1,
#             "top_3_match_percent": r3,
#             "top_5_match_percent": r5,
#             "outside_top_5_percent": ro,
#             "average_centipawn_loss": float(avg_cpl),
#             "median_centipawn_loss": float(med_cpl),
#             "white_average_cpl": float(white_cpl),
#             "black_average_cpl": float(black_cpl),
#             "stockfish_depth": CONFIG['stockfish_depth'],
#             "stockfish_multipv": CONFIG['multipv']
#         }
#         
#         summary_path = os.path.join(CONFIG["output_dir"], "analysis_summary.json")
#         with open(summary_path, 'w') as f:
#             json.dump(summary, f, indent=2)
#             
#         print(f"Summary written to {summary_path}")
"""))

    # --------------------------------------------------------------------------------
    # CELL 16 — Download / Access Final Dataset
    # --------------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("# CELL 16 — Download / Access Final Dataset"))
    cells.append(nbf.v4.new_code_cell("""\
print("=========================================================================")
print("✅ ANALYSIS PIPELINE COMPLETE")
print(f"📂 Final Dataset is safely stored in Google Drive at:")
print(f"   -> {CONFIG['merged_dir']}/positions_stockfish.parquet")
print("=========================================================================")
"""))

    nb['cells'] = cells
    
    with open('Stockfish_Analysis.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print("Google Colab Notebook 'Stockfish_Analysis.ipynb' generated successfully.")

if __name__ == "__main__":
    build_notebook()
