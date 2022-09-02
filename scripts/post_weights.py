"""Upload weights to github release"""

from pprint import pprint
import requests
import os
import dotenv
import argparse
from zipfile import ZipFile

dotenv.load_dotenv()


def create_release(tag, name, description, target="main"):
    """Create release"""
    token = os.environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}",
        "Content-Type": "application/zip"
    }
    url = "https://api.github.com/repos/ruhyadi/yolo3d-lightning/releases"
    payload = {
        "tag_name": tag,
        "target_commitish": target,
        "name": name,
        "body": description,
        "draft": True,
        "prerelease": False,
        "generate_release_notes": True,
    }
    print("[INFO] Creating release {}".format(tag))
    response = requests.post(url, json=payload, headers=headers)
    print("[INFO] Release created id: {}".format(response.json()["id"]))

    return response.json()


def post_assets(assets, release_id):
    """Post assets to release"""
    token = os.environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}",
        "Content-Type": "application/zip"
    }
    for asset in assets:
        asset_path = os.path.join(os.getcwd(), asset)
        with ZipFile(f"{asset_path}.zip", "w") as zip_file:
            zip_file.write(asset)
        asset_path = f"{asset_path}.zip"
        filename = asset_path.split("/")[-1]
        url = (
            "https://uploads.github.com/repos/ruhyadi/yolo3d-lightning/releases/"
            + str(release_id)
            + f"/assets?name={filename}"
        )
        print("[INFO] Uploading {}".format(filename))
        response = requests.post(url, files={"name": open(asset_path, "rb")}, headers=headers)
        pprint(response.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload weights to github release")
    parser.add_argument("--tag", type=str, default="v0.1", help="tag name")
    parser.add_argument("--name", type=str, default="Release v0.1", help="release name")
    parser.add_argument("--description", type=str, default="v0.1", help="release description")
    parser.add_argument("--assets", type=tuple, default=["weights/detector_yolov5s.pt", "weights/regressor_resnet18.pt"], help="directory to save weights",)
    args = parser.parse_args()

    release_id = create_release(args.tag, args.name, args.description)["id"]
    post_assets(args.assets, release_id)
