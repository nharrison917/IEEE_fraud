# -*- coding: utf-8 -*-
"""
Verify the chronological split looks sensible before proceeding with analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import load_data, make_split

df = load_data()
train, val, test = make_split(df)
