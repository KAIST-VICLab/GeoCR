"""``import export_rgbvariant`` in the comparison-method adapters resolves to ``geocr.data.exporters.export_rgbvariant``."""
import sys

from geocr.data.exporters import export_rgbvariant

sys.modules[__name__] = export_rgbvariant
