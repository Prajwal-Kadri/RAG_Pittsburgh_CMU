"""Merge source page files by URL; later inputs replace earlier copies."""
import argparse

from .common import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    pages = {}
    for filename in args.inputs:
        for page in read_jsonl(filename):
            pages[page["url"]] = page
    write_jsonl(pages.values(), args.out)
    print(f"Saved {len(pages)} unique source pages to {args.out}")


if __name__ == "__main__":
    main()
