"""Render every PlantUML diagram in this folder to an image.

Finds all ``*.puml`` files next to this script and renders each one with
PlantUML. Two backends are supported and tried in this order:

1. **Local Java** -- if ``java`` runs and ``plantuml.jar`` is present (it is
   downloaded once into this folder if missing). Fully offline after the first
   download.
2. **PlantUML web server** -- a fallback used when Java is unavailable. The
   diagram text is encoded and fetched from ``www.plantuml.com``; this needs
   network access and sends the diagram source to that public server.

Usage
-----
    python docs/diagrams/render_diagrams.py                 # PNG, auto backend
    python docs/diagrams/render_diagrams.py --format svg
    python docs/diagrams/render_diagrams.py --backend web   # force web server
    python docs/diagrams/render_diagrams.py --jar /path/to/plantuml.jar
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
import zlib
from pathlib import Path

DIAGRAMS_DIR = Path(__file__).resolve().parent
JAR_NAME = "plantuml.jar"
# Pinned release so local renders are reproducible.
JAR_URL = "https://github.com/plantuml/plantuml/releases/download/v1.2024.7/plantuml-1.2024.7.jar"
WEB_BASE = "https://www.plantuml.com/plantuml"


# --------------------------------------------------------------------------- #
# Local Java backend
# --------------------------------------------------------------------------- #
def java_available() -> bool:
    try:
        subprocess.run(["java", "-version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def ensure_jar(jar: Path) -> Path:
    if jar.exists():
        return jar
    print(f"plantuml.jar not found, downloading from {JAR_URL} ...")
    try:
        urllib.request.urlretrieve(JAR_URL, jar)
    except Exception as exc:  # noqa: BLE001 - surface any download failure clearly
        sys.exit(f"error: failed to download plantuml.jar: {exc}")
    print(f"saved {jar}")
    return jar


def render_local(puml: Path, jar: Path, fmt: str) -> None:
    print(f"[java] {puml.name} -> {puml.stem}.{fmt}")
    subprocess.run(
        ["java", "-jar", str(jar), f"-t{fmt}", "-o", str(DIAGRAMS_DIR), str(puml)],
        check=True,
    )


# --------------------------------------------------------------------------- #
# PlantUML web-server backend
# --------------------------------------------------------------------------- #
# PlantUML uses a custom base64 alphabet over raw-DEFLATE-compressed text.
_PLANTUML_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"


def _encode_plantuml(text: str) -> str:
    compressed = zlib.compress(text.encode("utf-8"), 9)[2:-4]  # strip zlib header/checksum
    out = []
    for i in range(0, len(compressed), 3):
        chunk = compressed[i:i + 3]
        b = chunk + bytes(3 - len(chunk))
        n = (b[0] << 16) | (b[1] << 8) | b[2]
        for shift in (18, 12, 6, 0):
            out.append(_PLANTUML_ALPHABET[(n >> shift) & 0x3F])
    return "".join(out)


def render_web(puml: Path, fmt: str) -> None:
    encoded = _encode_plantuml(puml.read_text(encoding="utf-8"))
    url = f"{WEB_BASE}/{fmt}/{encoded}"
    out_path = DIAGRAMS_DIR / f"{puml.stem}.{fmt}"
    print(f"[web]  {puml.name} -> {out_path.name}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (render_diagrams.py)"})
    try:
        with urllib.request.urlopen(request) as resp:
            out_path.write_bytes(resp.read())
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"error: web render failed for {puml.name}: {exc}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="png", choices=["png", "svg", "eps", "pdf"],
                        help="output image format (default: png)")
    parser.add_argument("--backend", default="auto", choices=["auto", "java", "web"],
                        help="rendering backend (default: auto -> java if available, else web)")
    parser.add_argument("--jar", type=Path, default=DIAGRAMS_DIR / JAR_NAME,
                        help="path to plantuml.jar (downloaded if missing)")
    args = parser.parse_args()

    puml_files = sorted(DIAGRAMS_DIR.glob("*.puml"))
    if not puml_files:
        sys.exit(f"no .puml files found in {DIAGRAMS_DIR}")

    backend = args.backend
    if backend == "auto":
        backend = "java" if java_available() else "web"
    if backend == "java" and not java_available():
        sys.exit("error: --backend java requested but no Java runtime is available.")

    if backend == "java":
        jar = ensure_jar(args.jar)
        for puml in puml_files:
            render_local(puml, jar, args.format)
    else:
        if args.format in {"eps", "pdf"}:
            sys.exit(f"error: web backend cannot produce {args.format}; use png or svg.")
        print("using PlantUML web server (no local Java runtime found)")
        for puml in puml_files:
            render_web(puml, args.format)

    print(f"done: rendered {len(puml_files)} diagram(s) to {DIAGRAMS_DIR}")


if __name__ == "__main__":
    main()
