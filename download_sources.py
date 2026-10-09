import argparse
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LITHOBENCH = "shelljane/lithobench/9c74e82218e377eaf6d02d113fc1ce6e36c92aa6"
OPENILT = "OpenOPC/OpenILT/dabb97c6ca3dfd159362e48273c436444c77353b"
OPENROAD = "The-OpenROAD-Project/OpenROAD-flow-scripts/e54722b45e683cd3f2a7506cb50596df8e0dc5cd"
SOURCES = [
    (LITHOBENCH, "kernel/kernels/focus.pt", "lithobench/kernel/kernels/focus.pt"),
    (LITHOBENCH, "kernel/kernels/defocus.pt", "lithobench/kernel/kernels/defocus.pt"),
    (LITHOBENCH, "kernel/scales/focus.pt", "lithobench/kernel/scales/focus.pt"),
    (LITHOBENCH, "kernel/scales/defocus.pt", "lithobench/kernel/scales/defocus.pt"),
    (LITHOBENCH, "config/lithosimple.txt", "lithobench/config/lithosimple.txt"),
    (LITHOBENCH, "LICENSE", "lithobench/LICENSE"),
    (OPENILT, "benchmark/gcd_45nm.gds", "openilt/gcd_45nm.gds"),
    (OPENILT, "LICENSE", "openilt/LICENSE"),
    (
        OPENROAD,
        "flow/platforms/nangate45/gds/NangateOpenCellLibrary.gds",
        "nangate45/NangateOpenCellLibrary.gds",
    ),
    (OPENROAD, "flow/platforms/nangate45/LICENSE", "nangate45/LICENSE"),
    (OPENROAD, "flow/platforms/nangate45/FreePDK45.lyt", "nangate45/FreePDK45.lyt"),
]


def download_sources(optics_only=False):
    for revision, source, relative in SOURCES:
        if optics_only and not relative.startswith("lithobench/"):
            continue
        destination = ROOT / "vendor" / relative
        if destination.exists():
            if destination.stat().st_size == 0:
                raise ValueError(f"Empty downloaded file: {destination}")
            print(f"Using {destination.relative_to(ROOT)}", flush=True)
            continue
        url = f"https://raw.githubusercontent.com/{revision}/{source}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "RCPWV"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if temporary.stat().st_size == 0:
                raise ValueError(f"Empty response: {url}")
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"Downloaded {destination.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download the pinned optical kernels and public GDS sources."
    )
    parser.add_argument(
        "--optics-only",
        action="store_true",
        help="Download only optical kernels, configuration and license.",
    )
    args = parser.parse_args()
    download_sources(args.optics_only)


if __name__ == "__main__":
    main()
