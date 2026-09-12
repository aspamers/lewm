"""Read public dataset metadata, without downloading large archives."""
import concurrent.futures
import json
import urllib.request


def read(name):
    url = f"https://huggingface.co/api/datasets/{name}/tree/main?recursive=true"
    try:
        with urllib.request.urlopen(url) as response:
            return name, json.load(response)
    except Exception as exc:
        return name, {"error": str(exc)}


if __name__ == "__main__":
    names = [f"{author}/lewm-{env}" for author in ("quentinll", "galilai-group")
             for env in ("tworooms", "pusht", "cube", "reacher")]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        print(json.dumps(dict(pool.map(read, names)), indent=2))
