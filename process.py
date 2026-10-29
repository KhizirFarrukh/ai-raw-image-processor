"""Entry point: python process.py <raw files/folders> -o output"""

import sys

from rawai.cli import main

if __name__ == "__main__":
    sys.exit(main())
