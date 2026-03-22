from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

url = "https://dl.dropboxusercontent.com/scl/fi/y8wvktb5lefnozak13aru/mati.csv?rlkey=hl3f7wpwe6ruadgx4v0cecya0"
output_path = DATASETS_DIR / "mati.csv"

print("Downloading dataset...")
urllib.request.urlretrieve(url, output_path)
print(f"Saved as {output_path}")