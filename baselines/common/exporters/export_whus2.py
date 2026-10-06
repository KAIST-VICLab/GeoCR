"""``import export_whus2`` in the comparison-method adapters resolves to ``geocr.data.exporters.export_whus2``."""
import sys

from geocr.data.exporters import export_whus2

sys.modules[__name__] = export_whus2
