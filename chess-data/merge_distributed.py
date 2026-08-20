import os
import glob
import pyarrow.parquet as pq
import pyarrow as pa
import argparse

def merge_chunks(chunks_dir, output_file):
    print(f"Scanning for chunks in: {chunks_dir}")
    chunks = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.parquet")))
    
    if not chunks:
        print("No completed chunks found.")
        return
        
    print(f"Found {len(chunks)} chunks. Reading metadata...")
    tables = []
    total_rows = 0
    
    for f in chunks:
        try:
            import pandas as pd
            df = pd.read_parquet(f)
            
            if "your_move_rank" in df.columns:
                df["your_move_rank"] = df["your_move_rank"].astype(str)
                df.loc[df["your_move_rank"] == "0", "your_move_rank"] = ">5"
                df.loc[df["your_move_rank"] == "0.0", "your_move_rank"] = ">5"
                
            for col in df.columns:
                if "eval_mate" in col:
                    df[col] = df[col].astype(float)
                    
            t = pa.Table.from_pandas(df)
            tables.append(t)
            total_rows += t.num_rows
        except Exception as e:
            print(f"Error reading chunk {f}: {e}")
            
    print(f"Merging {total_rows} rows from {len(tables)} chunks...")
    full_table = pa.concat_tables(tables, promote_options="permissive")
    
    print(f"Writing final dataset to {output_file}...")
    pq.write_table(full_table, output_file)
    
    print("Done! 🎉")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge distributed parquet chunks")
    parser.add_argument("--chunks-dir", required=True, help="Path to folder containing chunk_XXXXXX.parquet files")
    parser.add_argument("--output", default="data/analysis/positions_stockfish.parquet", help="Path to output final dataset")
    
    args = parser.parse_args()
    merge_chunks(args.chunks_dir, args.output)
