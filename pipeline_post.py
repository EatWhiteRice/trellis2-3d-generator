#!/usr/bin/env python3
"""Legacy CLI entry point; implementation lives in post_process.py."""
from post_process import calibrate_colors, extract_foreground_stats, geometry_smooth, main

if __name__ == '__main__':
    main()
