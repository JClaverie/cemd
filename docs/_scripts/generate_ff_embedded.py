# docs/_scripts/generate_ff_embedded.py
"""
Generate HTML with embedded JSON data.
"""

import json
from pathlib import Path

# Chemins
JSON_PATH = Path(__file__).parent.parent / "_static" / "ff_data.json"
HTML_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "_static" / "ff_viewer_template.html"
)
OUTPUT_PATH = Path(__file__).parent.parent / "_static" / "ff_viewer.html"


def generate_embedded_html():
    """Generate HTML with embedded JSON data."""

    if not JSON_PATH.exists():
        print(f"❌ JSON not found: {JSON_PATH}")
        return

    # Lire le JSON
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Lire le template HTML
    with open(HTML_TEMPLATE_PATH, encoding="utf-8") as f:
        html = f.read()

    # Remplacer EMBEDDED_DATA
    import re

    pattern = r"const EMBEDDED_DATA = \{.*?\};"
    replacement = f"const EMBEDDED_DATA = {json.dumps(data, indent=2)};"
    html = re.sub(pattern, replacement, html, flags=re.DOTALL)

    # Écrire le HTML généré
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated: {OUTPUT_PATH}")
    print(f"   Size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    generate_embedded_html()
