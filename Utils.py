import os
import requests

def download_file_if_not_exists(local_path, remote_url):
    if os.path.exists(local_path):
        return
    r = requests.get(remote_url)
    with open(local_path, "wb") as f:
        f.write(r.content)
