#!/usr/bin/env python3
"""arxiv Deep Reader — Download and extract full text from arxiv papers."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import requests


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/papers")


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def normalize_arxiv_id(input_str):
    """Extract arxiv ID from various formats (URL, ID, etc.)."""
    patterns = [
        r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"^(\d{4}\.\d{4,5}(?:v\d+)?)$",
    ]
    for p in patterns:
        m = re.search(p, input_str)
        if m:
            return m.group(1)
    return input_str.strip()


def get_metadata(arxiv_id):
    """Fetch paper metadata from arxiv API."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text

    def extract(tag):
        m = re.search(f"<{tag}[^>]*>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else None

    # Extract authors
    authors = re.findall(r"<name>(.*?)</name>", text)

    return {
        "id": arxiv_id,
        "title": extract("title"),
        "summary": extract("summary"),
        "authors": authors,
        "published": extract("published"),
        "updated": extract("updated"),
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
    }


def download_pdf(arxiv_id):
    """Download PDF to local cache."""
    ensure_cache_dir()
    pdf_path = os.path.join(CACHE_DIR, f"{arxiv_id.replace('/', '_')}.pdf")

    if os.path.exists(pdf_path):
        return pdf_path

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"Downloading {url}...", file=sys.stderr)
    resp = requests.get(url, timeout=60, headers={"User-Agent": "AGI-Research-Tool/1.0"})
    resp.raise_for_status()

    with open(pdf_path, "wb") as f:
        f.write(resp.content)

    print(f"Saved to {pdf_path}", file=sys.stderr)
    return pdf_path


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using python or system tools."""
    # Try using textutil (macOS native)
    try:
        result = subprocess.run(
            ["mdimport", "-d2", pdf_path],
            capture_output=True, text=True, timeout=30
        )
    except Exception:
        pass

    # Try using python-based extraction with subprocess
    # First try pdftotext if available
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except FileNotFoundError:
        pass

    # Fallback: use strings + cleanup for basic extraction
    try:
        result = subprocess.run(
            ["strings", pdf_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            # Basic cleanup of strings output
            lines = result.stdout.split("\n")
            # Filter out binary garbage, keep readable text
            readable = [l for l in lines if len(l) > 3 and any(c.isalpha() for c in l)]
            return "\n".join(readable)
    except Exception:
        pass

    return "[Could not extract text. Use PDF Viewer MCP tool to read: " + pdf_path + "]"


def search_papers(query, max_results=10):
    """Search arxiv for papers matching a query."""
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    for attempt in range(3):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = 3 * (attempt + 1)
            print(f"Rate limited, waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        resp.raise_for_status()

    entries = re.findall(r"<entry>(.*?)</entry>", resp.text, re.DOTALL)
    results = []
    for entry in entries:
        def extract(tag):
            m = re.search(f"<{tag}[^>]*>(.*?)</{tag}>", entry, re.DOTALL)
            return m.group(1).strip() if m else None

        arxiv_id_match = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)", entry)
        arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else extract("id")
        authors = re.findall(r"<name>(.*?)</name>", entry)

        results.append({
            "id": arxiv_id,
            "title": extract("title"),
            "summary": extract("summary"),
            "authors": authors[:5],
            "published": extract("published"),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="arxiv Deep Reader")
    sub = parser.add_subparsers(dest="command")

    # Fetch command
    fetch = sub.add_parser("fetch", help="Download and extract a paper")
    fetch.add_argument("paper_id", help="arxiv ID or URL")
    fetch.add_argument("--metadata-only", action="store_true", help="Only fetch metadata")
    fetch.add_argument("--json", action="store_true", help="Output as JSON")

    # Search command
    search = sub.add_parser("search", help="Search for papers")
    search.add_argument("query", help="Search query")
    search.add_argument("-n", "--max-results", type=int, default=10)
    search.add_argument("--json", action="store_true", help="Output as JSON")

    # List command
    lst = sub.add_parser("list", help="List cached papers")

    args = parser.parse_args()

    if args.command == "fetch":
        arxiv_id = normalize_arxiv_id(args.paper_id)
        meta = get_metadata(arxiv_id)

        if args.metadata_only:
            if args.json:
                print(json.dumps(meta, indent=2))
            else:
                print(f"Title:     {meta['title']}")
                print(f"Authors:   {', '.join(meta['authors'])}")
                print(f"Published: {meta['published']}")
                print(f"URL:       {meta['abs_url']}")
                print(f"\nAbstract:\n{meta['summary']}")
            return

        pdf_path = download_pdf(arxiv_id)
        text = extract_text_from_pdf(pdf_path)

        if args.json:
            meta["text"] = text
            meta["local_pdf"] = pdf_path
            print(json.dumps(meta, indent=2))
        else:
            print(f"=== {meta['title']} ===")
            print(f"Authors: {', '.join(meta['authors'])}")
            print(f"PDF: {pdf_path}")
            print(f"{'=' * 60}\n")
            print(text)

    elif args.command == "search":
        results = search_papers(args.query, args.max_results)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for i, r in enumerate(results, 1):
                print(f"\n[{i}] {r['title']}")
                print(f"    ID: {r['id']}")
                print(f"    Authors: {', '.join(r['authors'][:3])}")
                print(f"    Published: {r['published']}")
                summary = r['summary'][:200] + "..." if r['summary'] and len(r['summary']) > 200 else r['summary']
                print(f"    {summary}")

    elif args.command == "list":
        ensure_cache_dir()
        files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pdf")]
        if files:
            print(f"Cached papers in {CACHE_DIR}:")
            for f in sorted(files):
                size = os.path.getsize(os.path.join(CACHE_DIR, f))
                print(f"  {f} ({size / 1024:.0f} KB)")
        else:
            print("No cached papers.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
