"""Download pinned official archives with SHA256 verification and safe extraction."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

import zstandard

DATA = {
    "tworoom": ("tworooms", "6903a2de048b13819d812da0b4dd661290bc01e4", "tworoom.tar.zst",
                3425937909, "494b1a02f0765cd9a0d9daf1786c419ced1009977fc45d01e3158932f8d080ca", "tworoom.h5"),
    "pusht": ("pusht", "655cd446b9929369d7d406001da85c15d1457850", "pusht_expert_train.h5.zst",
              13136247974, "7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318", "pusht_expert_train.h5"),
    "cube": ("cube", "02a19a67a0dc8c9d6215f89c19e0a597691e152a", "cube_single_expert.tar.zst",
             46184624478, "3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600", "ogbench/cube_single_expert.h5"),
    "reacher": ("reacher", "e70a080d0d04c6072123c9ebd343acf7fff28dbf", "reacher.tar.zst",
                23750614946, "4ff2385e49712caa89f21b8e0a246e2614b621d3f22cf2d1224d845e879a1cc2", "reacher.h5"),
}


def prepare(name, root):
    repo, revision, filename, size, sha, target = DATA[name]
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / filename
    partial = downloads / (filename + ".part")
    destination = root / "datasets" / target
    status_path = root / (name + "-download.json")
    def status(stage, **kwargs):
        status_path.write_text(json.dumps({"environment": name, "stage": stage,
                                          "updated_at": time.time(), **kwargs}, indent=2))
    try:
        marker = destination.with_suffix(".verified.json")
        if destination.exists() and marker.exists():
            status("ready", path=str(destination), bytes=destination.stat().st_size)
            return
        if not archive.exists():
            for attempt in range(5):
                start = partial.stat().st_size if partial.exists() else 0
                if start == size:
                    break
                url = f"https://huggingface.co/datasets/quentinll/lewm-{repo}/resolve/{revision}/{filename}?download=true"
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-", "User-Agent": "lewm-benchmark"})
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        if start and response.status != 206:
                            raise RuntimeError("Server ignored range; refusing to append duplicate data")
                        with partial.open("ab") as output:
                            last = 0
                            while block := response.read(8*1024*1024):
                                output.write(block)
                                start += len(block)
                                if time.time()-last > 10:
                                    status("downloading", received=start, total=size)
                                    last = time.time()
                    if start != size:
                        raise RuntimeError(f"Expected {size} bytes, got {start}")
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2)
            status("verifying", bytes=size)
            with partial.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != sha:
                raise RuntimeError("Archive SHA256 mismatch")
            partial.rename(archive)
        else:
            with archive.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                    raise RuntimeError("Existing archive SHA256 mismatch")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(".h5.part")
        status("extracting", destination=str(destination))
        written = 0
        def copy_checked(source, output):
            nonlocal written
            last = 0
            while block := source.read(8*1024*1024):
                if shutil.disk_usage(root).free < 30*1024**3:
                    raise RuntimeError("Extraction stopped to preserve 30 GiB free disk")
                output.write(block)
                written += len(block)
                if time.time()-last > 10:
                    status("extracting", written=written, destination=str(destination))
                    last = time.time()
        with archive.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as decompressed:
            if filename.endswith(".tar.zst"):
                found = False
                with tarfile.open(fileobj=decompressed, mode="r|") as tar:
                    for member in tar:
                        if member.isfile() and member.name.endswith((".h5", ".hdf5")):
                            if found:
                                raise RuntimeError("More than one HDF5 file; explicit mapping required")
                            if member.size > shutil.disk_usage(root).free - 30*1024**3:
                                raise RuntimeError("Insufficient disk for archive member")
                            with tar.extractfile(member) as content, temp.open("wb") as output:
                                copy_checked(content, output)
                            found = True
                    if not found:
                        raise RuntimeError("Archive has no HDF5 file")
            else:
                with temp.open("wb") as output:
                    copy_checked(decompressed, output)
        import h5py
        import hdf5plugin  # noqa: F401
        with h5py.File(temp, "r") as dataset:
            columns = {key: list(value.shape) for key, value in dataset.items() if hasattr(value, "shape")}
        if destination.exists():
            raise FileExistsError(destination)
        temp.rename(destination)
        marker.write_text(json.dumps({"repository": f"quentinll/lewm-{repo}", "revision": revision,
                                     "archive_sha256": sha, "columns": columns, "bytes": written}, indent=2))
        status("ready", path=str(destination), bytes=written, columns=columns)
    except Exception as exc:
        status("error", error=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--environments", nargs="+", choices=DATA, default=list(DATA))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda name: prepare(name, args.root), args.environments))
