"""
Render all .puml files in this directory to PNG images.
Uses the public PlantUML server — no Java or local install needed.
Requires: pip install requests
"""

import zlib
import requests
from pathlib import Path


# ── PlantUML encoding ────────────────────────────────────────────────────────

def _encode6bit(b: int) -> str:
    if b < 10:
        return chr(48 + b)
    b -= 10
    if b < 26:
        return chr(65 + b)
    b -= 26
    if b < 26:
        return chr(97 + b)
    b -= 26
    if b == 0:
        return "-"
    if b == 1:
        return "_"
    return "?"


def _append3bytes(b1: int, b2: int, b3: int) -> str:
    c1 = b1 >> 2
    c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
    c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
    c4 = b3 & 0x3F
    return _encode6bit(c1) + _encode6bit(c2) + _encode6bit(c3) + _encode6bit(c4)


def _encode_bytes(data: bytes) -> str:
    result = ""
    i = 0
    while i + 2 < len(data):
        result += _append3bytes(data[i], data[i + 1], data[i + 2])
        i += 3
    remaining = len(data) - i
    if remaining == 1:
        result += _append3bytes(data[i], 0, 0)
    elif remaining == 2:
        result += _append3bytes(data[i], data[i + 1], 0)
    return result


def plantuml_encode(source: str) -> str:
    """Compress and encode PlantUML source for use in the server URL."""
    compressed = zlib.compress(source.encode("utf-8"), level=9)
    compressed = compressed[2:-4]  # strip zlib header and Adler-32 checksum
    return _encode_bytes(compressed)


# ── Rendering ────────────────────────────────────────────────────────────────

SERVER = "https://www.plantuml.com/plantuml/png"


def render(puml_path: Path, out_dir: Path) -> Path:
    source = puml_path.read_text(encoding="utf-8")
    encoded = plantuml_encode(source)
    url = f"{SERVER}/{encoded}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    out_path = out_dir / puml_path.with_suffix(".png").name
    out_path.write_bytes(response.content)
    return out_path


def main():
    here = Path(__file__).parent
    puml_files = sorted(here.glob("*.puml"))

    if not puml_files:
        print("No .puml files found.")
        return

    print(f"Rendering {len(puml_files)} diagram(s)...\n")
    for puml in puml_files:
        try:
            out = render(puml, here)
            print(f"  OK  {puml.name} -> {out.name}")
        except Exception as e:
            print(f"  ERR {puml.name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
