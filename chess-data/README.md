# Chess.com Game Data Downloader

This is a Python tool that downloads a user's Chess.com games in PGN format and organizes them locally. It builds a structured dataset for machine learning or analysis projects.

## Setup

1. **Install Python 3.11+** if you haven't already.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config.json` in the root directory to set the username you want to download games for:

```json
{
  "username": "yh_am_in",
  "output_directory": "data"
}
```

## Running the Downloader

To download the games and generate the dataset, run:

```bash
python fetch_games.py
```

### Incremental Downloading
The downloader automatically checks which monthly archives have already been downloaded (by looking in the `data/games/USERNAME/raw/` folder). It will only download new or missing months, saving you time and API calls.

If you wish to force a complete redownload of all archives, run:

```bash
python fetch_games.py --force
```

## Output Structure

The tool creates the following directory structure inside the configured `output_directory` (default is `data`):

```text
data/
├── games/
│   └── YOUR_USERNAME/
│       ├── raw/
│       │   ├── 2024-01.pgn
│       │   ├── 2024-02.pgn
│       │   └── ...
│       └── combined/
│           └── all_games.pgn
└── processed/
    └── games.csv
```

- **raw/**: Contains the untouched, raw PGN text directly from Chess.com, separated by month.
- **combined/all_games.pgn**: A single large PGN file containing all games.
- **processed/games.csv**: A structured dataset where each row represents one game with all its metadata (e.g., date, ratings, time control, eco) and the raw PGN move text.

## Regenerating the Dataset

If you ever need to regenerate `games.csv` and `all_games.pgn` without redownloading the data (for example, if you change the processing code), you can delete `data/processed/games.csv` and run the script normally, or you can run `python fetch_games.py --force`.
