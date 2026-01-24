import os
import sys
sys.path.append('../../src/')

from datetime import datetime, timezone

def utc() -> str:
    return datetime.now(timezone.utc).isoformat()