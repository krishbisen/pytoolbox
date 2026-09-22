"""Command-line interface for PyToolbox."""

from __future__ import annotations

import argparse

from .utils import chunked, clamp, is_palindrome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pytoolbox",
        description="Small Python utilities for everyday programming tasks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clamp_parser = subparsers.add_parser("clamp", help="constrain a number to a range")
    clamp_parser.add_argument("value", type=float)
    clamp_parser.add_argument("minimum", type=float)
    clamp_parser.add_argument("maximum", type=float)

    palindrome_parser = subparsers.add_parser("palindrome", help="check text")
    palindrome_parser.add_argument("text")

    chunk_parser = subparsers.add_parser("chunk", help="split values into chunks")
    chunk_parser.add_argument("values", nargs="+")
    chunk_parser.add_argument("--size", type=int, required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "clamp":
        print(clamp(args.value, args.minimum, args.maximum))
    elif args.command == "palindrome":
        print(is_palindrome(args.text))
    elif args.command == "chunk":
        print(list(chunked(args.values, args.size)))


if __name__ == "__main__":
    main()
