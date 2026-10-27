"""Convert the trained TF-IDF + LogisticRegression pipeline into an ONNX graph.

Why this module exists instead of a one-line ``skl2onnx.convert_sklearn`` call:
the stock skl2onnx converter is NOT numerically faithful for our vectorizer
configuration (verified against ``models/model.joblib`` with skl2onnx 1.20 /
scikit-learn 1.9):

1. ``stop_words="english"`` is silently dropped. skl2onnx reads the
   ``stop_words_`` attribute, which recent scikit-learn no longer sets, so the
   ONNX graph keeps stop words and builds bigrams across them
   (``"pain to the arm"`` -> ``"pain to"`` instead of sklearn's ``"pain arm"``).
   Its ``StringNormalizer`` would also run on whole documents, where a
   stop-word list can never match a single token.
2. The default sklearn token pattern ``(?u)\\b\\w\\w+\\b`` is replaced by
   ``[a-zA-Z0-9_]+`` (keeps 1-char tokens, drops non-ASCII letters).
3. Some bigrams whose words are not unigrams in the vocabulary
   (e.g. ``"bismuth subsalicylate"``) are missing from the n-gram pool.
4. The ``StringNormalizer`` defaults to the ``en_US.UTF-8`` locale, which
   does not exist in slim Docker images, so the session fails to load.

On the 2,888-text test split the stock export disagreed with sklearn on
probabilities by up to 0.136 and on ~1.2% of the labels. We therefore build
the text-featurisation subgraph by hand, reproducing sklearn's analyzer order
(lowercase -> tokenize -> drop stop words -> n-grams), and reuse skl2onnx only
for the purely numeric tail (IDF weighting, L2 norm, LinearClassifier).
"""

from __future__ import annotations

import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

INPUT_NAME = "texto"
# Metadata key read by src/api/backends.py so the serving layer never has to
# hardcode class names or parse the graph.
CLASSES_METADATA_KEY = "vitorium_classes"

# sklearn's default ``(?u)\b\w\w+\b``: a "word" is a run of >= 2 Unicode
# letters, digits or underscores. RE2 (used by onnxruntime's Tokenizer) has no
# Unicode ``\w``, so the class is spelled out explicitly.
SKLEARN_DEFAULT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"
RE2_TOKEN_PATTERN = r"[\p{L}\p{N}_]{2,}"

# C.UTF-8 ships with glibc >= 2.35 (Debian bookworm, i.e. python:3.11-slim)
# and lowercases non-ASCII letters like Python's str.lower(); plain "C" does not.
NORMALIZER_LOCALE = "C.UTF-8"

_FEATURIZER_OPS = {"Reshape", "StringNormalizer", "Tokenizer", "Flatten", "TfIdfVectorizer"}


def _check_supported(vectorizer: TfidfVectorizer) -> None:
    """Fail loudly on vectorizer options the hand-built subgraph does not model."""
    unsupported = {
        "analyzer": vectorizer.analyzer != "word",
        "token_pattern": vectorizer.token_pattern != SKLEARN_DEFAULT_TOKEN_PATTERN,
        "tokenizer": vectorizer.tokenizer is not None,
        "preprocessor": vectorizer.preprocessor is not None,
        "strip_accents": vectorizer.strip_accents is not None,
        "binary": bool(vectorizer.binary),
    }
    bad = [name for name, flag in unsupported.items() if flag]
    if bad:
        raise NotImplementedError(f"ONNX export does not support TfidfVectorizer options: {bad}")


def _build_featurizer_body(vectorizer: TfidfVectorizer, pad_value: str) -> onnx.GraphProto:
    """Per-document subgraph: raw text [1] -> term counts [1, n_features].

    It runs inside a ``Scan`` because ``StringNormalizer`` only accepts a single
    row of tokens; the scan keeps the graph correct for any batch size.
    """
    vocab: dict[str, int] = vectorizer.vocabulary_
    min_n, max_n = vectorizer.ngram_range

    # ONNX TfIdfVectorizer wants the pool grouped by n-gram length, plus the
    # pool offset of each group and the output column of every n-gram.
    pool: list[str] = []
    ngram_counts: list[int] = []
    ngram_indexes: list[int] = []
    for n in range(min_n, max_n + 1):
        ngram_counts.append(len(pool))
        grams = sorted((term, col) for term, col in vocab.items() if len(term.split(" ")) == n)
        for term, col in grams:
            pool.extend(term.split(" "))
            ngram_indexes.append(col)
    if len(ngram_indexes) != len(vocab):
        raise RuntimeError("Vocabulary contains n-grams outside ngram_range; cannot export.")

    stop_words = sorted(vectorizer.get_stop_words() or [])
    nodes = [
        helper.make_node(
            "Tokenizer",
            ["doc"],
            ["tokens"],
            domain="com.microsoft",
            mark=0,
            mincharnum=1,
            pad_value=pad_value,
            tokenexp=RE2_TOKEN_PATTERN,
        ),
        # Appending one pad token guarantees StringNormalizer never sees an empty
        # row (text without tokens). The pad is not in the vocabulary, so it adds
        # no unigram, and any bigram it forms ("last_word <pad>") is not either.
        helper.make_node("Concat", ["tokens", "pad_token"], ["tokens_padded"], axis=1),
        helper.make_node(
            "StringNormalizer",
            ["tokens_padded"],
            ["tokens_clean"],
            case_change_action="LOWER" if vectorizer.lowercase else "NONE",
            is_case_sensitive=0 if vectorizer.lowercase else 1,
            locale=NORMALIZER_LOCALE,
            stopwords=stop_words,
        ),
        helper.make_node(
            "TfIdfVectorizer",
            ["tokens_clean"],
            ["counts"],
            mode="TF",
            min_gram_length=min_n,
            max_gram_length=max_n,
            max_skip_count=0,
            pool_strings=pool,
            ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes,
        ),
    ]
    return helper.make_graph(
        nodes,
        "vitorium_featurizer_row",
        [helper.make_tensor_value_info("doc", TensorProto.STRING, [1])],
        [helper.make_tensor_value_info("counts", TensorProto.FLOAT, [1, None])],
        initializer=[numpy_helper.from_array(np.array([[pad_value]], dtype=object), "pad_token")],
    )


def convert_pipeline_to_onnx(pipeline: Pipeline) -> onnx.ModelProto:
    """Return an ONNX model with sklearn-equivalent ``label`` and ``probabilities``.

    Input: ``texto`` string tensor of shape [N, 1]. Outputs: ``label`` [N] and
    ``probabilities`` [N, n_classes] in ``pipeline.classes_`` order. The class
    list is also stored in metadata (``vitorium_classes``) for the API.
    """
    vectorizer = pipeline.steps[0][1]
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError("Expected the first pipeline step to be a TfidfVectorizer.")
    _check_supported(vectorizer)

    model = convert_sklearn(
        pipeline,
        initial_types=[(INPUT_NAME, StringTensorType([None, 1]))],
        options={id(pipeline): {"zipmap": False}},
    )
    graph = model.graph

    tfidf_nodes = [n for n in graph.node if n.op_type == "TfIdfVectorizer"]
    if len(tfidf_nodes) != 1:
        raise RuntimeError("Expected exactly one TfIdfVectorizer node in the skl2onnx graph.")
    counts_name = tfidf_nodes[0].output[0]

    featurizer = [n for n in graph.node if n.op_type in _FEATURIZER_OPS]
    if {n.op_type for n in featurizer} != _FEATURIZER_OPS:
        raise RuntimeError("Unexpected skl2onnx graph layout; refusing to rewrite it.")
    tail = [n for n in graph.node if n.op_type not in _FEATURIZER_OPS]

    pad_value = "#"
    while pad_value in vectorizer.vocabulary_:
        pad_value += "#"

    new_nodes = [
        helper.make_node("Reshape", [INPUT_NAME, "vitorium_rows_shape"], ["vitorium_rows"]),
        helper.make_node(
            "Scan",
            ["vitorium_rows"],
            ["vitorium_counts_3d"],
            body=_build_featurizer_body(vectorizer, pad_value),
            num_scan_inputs=1,
        ),
        helper.make_node("Reshape", ["vitorium_counts_3d", "vitorium_counts_shape"], [counts_name]),
    ]
    del graph.node[:]
    graph.node.extend(new_nodes + tail)

    used_inputs = {name for node in graph.node for name in node.input}
    kept = [init for init in graph.initializer if init.name in used_inputs]
    del graph.initializer[:]
    graph.initializer.extend(
        kept
        + [
            numpy_helper.from_array(np.array([-1, 1], dtype=np.int64), "vitorium_rows_shape"),
            numpy_helper.from_array(np.array([0, -1], dtype=np.int64), "vitorium_counts_shape"),
        ]
    )

    classes = [str(c) for c in pipeline.classes_]
    helper.set_model_props(model, {CLASSES_METADATA_KEY: json.dumps(classes)})
    onnx.checker.check_model(model)
    return model
