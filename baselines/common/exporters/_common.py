"""``import _common`` in the comparison-method adapters resolves to ``geocr.data.exporters._common``."""
import sys

from geocr.data.exporters import _common

sys.modules[__name__] = _common
