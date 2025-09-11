#%%
from pathlib import Path
import dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import mapillary.interface as mly
import os, json, logging

# ---- config ----
FLAGS_PATH = "C:/Users/ANNIE CHEN/Box Sync/Personal/flags/"
dotenv.load_dotenv(".env")
mly.set_access_token(os.getenv("MLY_KEY"))
TIMEOUT = (5, 30)     

#%%
def make_session():
    sess = requests.Session()
    retry = Retry(total=5, backoff_factor=1.2,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "HEAD"])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
    sess.mount("https://", adapter); sess.mount("http://", adapter)
    sess.headers.update({"Authorization": f"OAuth {os.getenv('MLY_KEY')}"})
    return sess

session = make_session()

def chunked(iterable, n):
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            if chunk: yield chunk
            break
        yield chunk

def get_thumb_urls(image_ids):
    """Return dict {id: best_thumb_url} using Graph API /images."""
    out = {}
    # small batches avoid URL length issues; 50 is usually safe
    for batch in chunked(image_ids, 50):
        ids_param = ",".join(str(i) for i in batch)
        url = "https://graph.mapillary.com/images"
        params = {
            "image_ids": ids_param,
            "fields": "id,thumb_1024_url"
        }
        r = session.get(url, params=params, timeout=(5, 30))
        r.raise_for_status()
        data = r.json()
        for item in data.get("data", []):
            img_id = item["id"]
            thumb = item.get("thumb_1024_url")
            if thumb:
                out[img_id] = thumb
    return out

def download(url, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=(5, 60)) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(1024 * 64):
                if chunk:
                    f.write(chunk)
#%%
def main():
    with open(f"{FLAGS_PATH}metadata/images_in_grid_202025/nonempty_grid_metadata_202025.json", "r") as f:
        nonempty_grid_metadata = json.load(f)
    # set up logger
    logging.basicConfig(
        filename="C:\\Users\\ANNIE CHEN\\Desktop\\Flags\\logs\\download_images_logger.log",
        filemode='a', # append
        force=True,
        format="%(levelname)s | %(asctime)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.INFO # minimum level accepteds
    )
    #len(nonempty_grid_metadata) 
    #1203
    for grid in range(22, len(nonempty_grid_metadata)):
        # 1) collect IDs from your dict
        image_ids = [f["properties"]["id"] for f in nonempty_grid_metadata[grid]["features"]]
        logging.info(f"Found {len(image_ids)} images in non-empty grid {grid+1}/{len(nonempty_grid_metadata)}")
        # 2) get their thumbnail URLs (prefers 1024px )
        id2url = get_thumb_urls(image_ids)
        # 3) download to your folder scheme
        out_dir = Path(FLAGS_PATH) / "mapillary" / "202025"
        results = []
        for img_id in image_ids:
            logging.info(f"Processing image {img_id} in non-empty grid {grid+1}/{len(nonempty_grid_metadata)}")
            url = id2url.get(str(img_id)) or id2url.get(img_id)
            if not url:
                results.append((img_id, None, "no thumbnail url")); continue
            out_path = out_dir / f"mapillary-{img_id}.jpg"
            try:
                logging.info(f"Downloading {url} to {out_path}")
                download(url, out_path)
                results.append((img_id, str(out_path), "ok"))
            except Exception as e:
                results.append((img_id, None, str(e)))
    logging.info("Done!")
#%%
print(main())
#%%
count = 0
for grid in range(len(nonempty_grid_metadata)):
    print(grid)
    image_ids = [f["properties"]["id"] for f in nonempty_grid_metadata[grid]["features"]]
    n_imgs = len(image_ids)
    count += n_imgs
    if 302627214836098 in image_ids: # grid = 263 /1203
        print("found it")
        break
    else:
        continue
#%%