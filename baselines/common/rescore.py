"""``import rescore`` in the comparison-method adapters: the dataset registry and scorer of
``geocr.eval.score`` under the module name the adapters use."""
from geocr.eval.score import (  # noqa: F401
    CORPORA, IDENTITY_KEYS, check_protocol_identity, load_pred, metrics_one, read_split,
    resolve_corpus, sha256_file, sid_flat,
)
