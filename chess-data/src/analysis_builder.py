import os
import time
import glob
import math
import multiprocessing
import pandas as pd
import pyarrow.parquet as pq
import datetime

from src.mp_worker import init_worker, worker_job

class AnalysisBuilder:
    def __init__(self, data_dir: str, config: dict):
        self.data_dir = data_dir
        self.config = config
        
        self.positions_file = os.path.join(data_dir, "positions", "positions.parquet")
        self.analysis_dir = os.path.join(data_dir, "analysis")
        
        # New local chunks directory
        self.chunks_dir = os.path.join(self.analysis_dir, "local_chunks")
        os.makedirs(self.chunks_dir, exist_ok=True)
        
        self.error_log = os.path.join(data_dir, "..", "logs", "stockfish_errors.log")
        
        sf_conf = config.get("stockfish", {})
        self.depth = sf_conf.get("depth", 10)
        self.multipv = sf_conf.get("multipv", 5)
        self.threads = sf_conf.get("threads_per_worker", 1)  # 1 thread per worker is safest
        self.hash_mb = sf_conf.get("hash_mb", 128)
        self.batch_size = sf_conf.get("chunk_size", 1000)
        
        # Determine num workers
        import psutil
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        safe = max(1, physical - 1) if physical else max(1, logical - 1)
        self.num_workers = config.get("num_workers") or safe
        
        sf_path_raw = config.get("stockfish_path", "stockfish/stockfish-windows-x86-64-avx2.exe")
        if not os.path.isabs(sf_path_raw):
            self.stockfish_path = os.path.join(os.path.dirname(data_dir), sf_path_raw)
        else:
            self.stockfish_path = sf_path_raw
            
    def build(self, reverse: bool = True, limit: int = None):
        if not os.path.exists(self.positions_file):
            print(f"Dataset not found at {self.positions_file}")
            return
            
        print("Loading position dataset...")
        table = pq.read_table(self.positions_file)
        
        if limit:
            df_full = table.slice(0, limit).to_pandas()
        else:
            df_full = table.to_pandas()
            
        total_positions = len(df_full)
        total_chunks = math.ceil(total_positions / self.batch_size)
        
        # Check existing chunks
        existing = glob.glob(os.path.join(self.chunks_dir, "chunk_*.parquet"))
        completed_ids = set()
        for f in existing:
            base = os.path.basename(f)
            if base.startswith("chunk_") and base.endswith(".parquet"):
                try:
                    idx = int(base.split("_")[1].split(".")[0])
                    completed_ids.add(idx)
                except:
                    pass
                    
        print(f"Total chunks: {total_chunks} | Completed: {len(completed_ids)} | Remaining: {total_chunks - len(completed_ids)}")
        
        stats = {
            "total_positions": total_positions,
            "completed_positions": len(completed_ids) * self.batch_size,
            "rank_1": 0, "top_3": 0, "cpl_sum": 0, "outside": 0
        }
        
        start_time = time.time()
        positions_processed_this_run = 0
        
        # Decide direction
        chunk_range = reversed(range(total_chunks)) if reverse else range(total_chunks)
        
        print(f"Starting Distributed Analysis... (Reverse: {reverse})")
        print(f"Workers: {self.num_workers} | Stockfish Threads: {self.threads}")
        
        for chunk_id in chunk_range:
            if chunk_id in completed_ids:
                continue
                
            start_row = chunk_id * self.batch_size
            end_row = min(start_row + self.batch_size, total_positions)
            chunk_df = df_full.iloc[start_row:end_row].copy()
            
            try:
                # Prepare jobs
                jobs = []
                for idx, r in chunk_df.iterrows():
                    move_col = "your_move_uci" if "your_move_uci" in r else "your_move"
                    jobs.append((r['fen'], r[move_col], r['your_color'], self.depth, self.multipv))
                
                # Execute in parallel
                with multiprocessing.Pool(
                    self.num_workers, 
                    initializer=init_worker, 
                    initargs=(self.stockfish_path, self.threads, self.hash_mb)
                ) as pool:
                    results = pool.map(worker_job, jobs)
                    
                # Merge results back to dataframe
                for i, res in enumerate(results):
                    for k, v in res.items():
                        chunk_df.loc[chunk_df.index[i], k] = v
                    chunk_df.loc[chunk_df.index[i], "analysis_timestamp"] = str(datetime.datetime.now())
                    
                    # Track stats
                    if res.get("your_move_rank") == "1":
                        stats["rank_1"] += 1
                        stats["top_3"] += 1
                    elif res.get("your_move_rank") in ["2", "3"]:
                        stats["top_3"] += 1
                    elif res.get("your_move_rank") == ">5":
                        stats["outside"] += 1
                    stats["cpl_sum"] += res.get("centipawn_loss", 0)
                    
                # Save chunk atomically
                tmp_path = os.path.join(self.chunks_dir, f"chunk_{chunk_id:06d}.tmp.parquet")
                final_path = os.path.join(self.chunks_dir, f"chunk_{chunk_id:06d}.parquet")
                
                chunk_df.to_parquet(tmp_path, index=False)
                os.rename(tmp_path, final_path)
                
                n = len(chunk_df)
                positions_processed_this_run += n
                stats["completed_positions"] += n
                
                # Print Progress
                elapsed = time.time() - start_time
                speed = positions_processed_this_run / elapsed if elapsed > 0 else 0
                rem = total_positions - stats["completed_positions"]
                eta_m = (rem / speed) / 60 if speed > 0 else 0
                pct = stats["completed_positions"] / total_positions * 100
                
                r1_pct = (stats["rank_1"] / positions_processed_this_run * 100) if positions_processed_this_run else 0
                cpl_avg = (stats["cpl_sum"] / positions_processed_this_run) if positions_processed_this_run else 0
                
                print(f"\\n--- Chunk {chunk_id:06d} Complete ---")
                print(f"Progress: {pct:.2f}% | Speed: {speed:.1f} pos/sec | ETA: {eta_m:.1f} min")
                print(f"Best Move Match: {r1_pct:.1f}% | Avg CPL: {cpl_avg:.1f}")
                
            except Exception as e:
                print(f"Error processing chunk {chunk_id}: {e}")
                
        print("Analysis process finished.")
