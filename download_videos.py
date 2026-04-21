import os
import csv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

METADATA_FILE = 'webvid_metadata.csv'
DOWNLOAD_DIR = 'videos'
MAX_VIDEOS = 500
MAX_WORKERS = 10  # adjust based on your network speed

def download_video(videoid, url, pbar):
    try:
        file_path = os.path.join(DOWNLOAD_DIR, f"{videoid}.mp4")
        
        # Skip if already downloaded
        if os.path.exists(file_path):
            pbar.update(1)
            return True
        
        # Download the video
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        pbar.update(1)
        return True
    except Exception as e:
        # Silently fail for dead links, we can just download more if needed
        pbar.update(1)
        # print(f"Error downloading {videoid}: {e}")
        return False

def main():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        
    videos_to_download = []
    
    print(f"Reading {METADATA_FILE}...")
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(videos_to_download) >= MAX_VIDEOS:
                break
            
            videoid = row.get('videoid')
            url = row.get('contentUrl')
            
            if videoid and url:
                videos_to_download.append((videoid, url))
                
    if not videos_to_download:
        print("No videos found in the CSV. Please check the column names.")
        return
        
    print(f"Starting download of {len(videos_to_download)} videos...")
    
    successful_downloads = 0
    with tqdm(total=len(videos_to_download), desc="Downloading Videos") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            futures = [executor.submit(download_video, vid, url, pbar) for vid, url in videos_to_download]
            
            # Wait for them to complete
            for future in as_completed(futures):
                if future.result():
                    successful_downloads += 1
                    
    print(f"\nFinished! Successfully downloaded {successful_downloads}/{len(videos_to_download)} videos.")
    print(f"Videos are saved in the '{DOWNLOAD_DIR}/' directory.")

if __name__ == "__main__":
    main()
