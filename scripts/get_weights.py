"""Download pretrained weights from github release"""

from pprint import pprint
import requests
import os
import shutil
import argparse
from zipfile import ZipFile

def get_assets(tag):
    """Get release assets by tag name"""
    url = 'https://api.github.com/repos/ruhyadi/yolo3d-lightning/releases/tags/' + tag
    response = requests.get(url)
    return response.json()['assets']

def download_assets(assets, dir):
    """Download assets to dir"""
    for asset in assets:
        url = asset['browser_download_url']
        filename = asset['name']
        print('[INFO] Downloading {}'.format(filename))
        response = requests.get(url, stream=True)
        with open(os.path.join(dir, filename), 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        del response

        with ZipFile(os.path.join(dir, filename), 'r') as zip_file:
            zip_file.extractall(dir)
        os.remove(os.path.join(dir, filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download pretrained weights')
    parser.add_argument('--tag', type=str, default='v0.1', help='tag name')
    parser.add_argument('--dir', type=str, default='./', help='directory to save weights')
    args = parser.parse_args()

    assets = get_assets(args.tag)
    download_assets(assets, args.dir)
