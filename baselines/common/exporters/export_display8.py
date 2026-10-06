"""``import export_display8`` in the comparison-method adapters resolves to ``geocr.data.exporters.export_display8``."""
import sys

from geocr.data.exporters import export_display8

sys.modules[__name__] = export_display8
