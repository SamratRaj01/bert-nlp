"""Assignment 3 / Workstream A tests (handout section 18: data continuity, alignment, geometry, E)."""
import hashlib
import json
import math

import numpy as np
import pytest

from src.drift.audit import check_continuity, check_val_sample, load_records, summarize_condition
from src.drift.embeddings import SpanEmbedder, load_E, pool_pieces
from src.drift.geometry import EMPTY, VALID, ZERO_NORM, pair_metrics
from src.drift.paths import E_PATH, E_SHAPE
from src.drift.spans import noisy_span, reassemble, select_spans, split_chunks
from src.noise.grid import conditions, load_condition
from src.phase2_config import load_phase2_config


def _span(text, idx):
    ch = split_chunks(text)[idx]
    return {"span_id": f"r:{idx}", "record_id": "r", **ch}


# ---- alignment ------------------------------------------------------------------------------
TEXT = "The cat sat on a mat"


def test_substitution_keeps_span_and_offsets():
    s = _span(TEXT, 1)  # "cat"
    log = [{"op": "substitution", "index": 5, "source_char": "a", "replacement_char": "o"}]
    assert (s["span_id"], s["start"], s["end"], s["clean"]) == ("r:1", 4, 7, "cat")
    assert noisy_span(s, log) == "cot"
    assert noisy_span(_span(TEXT, 0), log) == "The"  # an edit in another span does not leak


def test_insertion_belongs_to_its_original_position():
    log = [{"op": "insertion", "index": 1, "source_char": None, "replacement_char": "x"}]
    assert noisy_span(_span(TEXT, 0), log) == "Thxe"


def test_fully_deleted_span_stays_an_explicit_empty_pair():
    s = _span(TEXT, 4)  # "a"
    log = [{"op": "deletion", "index": s["start"], "source_char": "a", "replacement_char": None}]
    assert noisy_span(s, log) == ""
    spans = select_spans([{"record_id": "r", "text": TEXT}])
    assert [x["span_id"] for x in spans] == [f"r:{i}" for i in range(6)]  # nothing reindexed


def test_chunks_without_a_letter_are_skipped_but_keep_their_index():
    spans = select_spans([{"record_id": "r", "text": "( 24 ) cat , @-@ dog"}])
    assert [(x["span_id"], x["clean"]) for x in spans] == [("r:3", "cat"), ("r:6", "dog")]


def test_limit_and_record_order():
    recs = [{"record_id": "a", "text": "one two"}, {"record_id": "b", "text": "three four"}]
    assert [x["span_id"] for x in select_spans(recs, limit=3)] == ["a:0", "a:1", "b:0"]


def test_fragmentation_into_more_subwords():
    from src.tokenizer.wrapper import TokenizerWrapper
    tok = TokenizerWrapper().tokenizer
    assert len(tok.tokenize("lobster")) == 1
    assert len(tok.tokenize("lobstxr")) > 1  # one substitution splits the word
    assert tok.tokenize("") == []


def test_reassembled_record_equals_saved_noisy_text():
    cfg = load_phase2_config()
    records = {r["record_id"]: r["text"] for r in load_records()}
    for cid in ("ins-p0.20-s11", "del-p0.20-s23", "sub-p0.10-s47"):
        for row in load_condition(cid)[:60]:
            assert reassemble(records[row["record_id"]], row["operation_log"]) == row["noisy_text"]


# ---- data continuity ------------------------------------------------------------------------
def test_ids_hashes_and_order_match_across_all_37_conditions():
    cfg = load_phase2_config()
    records = load_records()
    assert len(records) == 500
    rows = {c["condition_id"]: load_condition(c["condition_id"]) for c in conditions(cfg)}
    assert len(rows) == 37
    check_continuity(records, rows)
    check_val_sample()  # LF-normalized sha256 equals the one recorded in Phase 2


# ---- geometry -------------------------------------------------------------------------------
def test_identical_vectors():
    u = np.array([3.0, 4.0, 0.0])
    cos, dist, status = pair_metrics(u, u.copy())
    assert status == VALID and math.isclose(cos, 1.0, abs_tol=1e-12) and dist == 0.0


def test_orthogonal_unit_vectors():
    cos, dist, status = pair_metrics(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert status == VALID and math.isclose(cos, 0.0, abs_tol=1e-12)
    assert math.isclose(dist, math.sqrt(2.0), rel_tol=1e-12)


def test_undefined_pairs_are_flagged_not_imputed():
    u = np.array([1.0, 2.0])
    assert pair_metrics(u, None) == (None, None, EMPTY)
    assert pair_metrics(np.zeros(2), u) == (None, None, ZERO_NORM)
    assert pair_metrics(u, np.zeros(2)) == (None, None, ZERO_NORM)


def test_condition_summary_counts_undefined_and_averages_valid_only():
    base = {"n_clean_pieces": 1, "n_noisy_pieces": 1, "noisy_unk": 0, "span_med": 0,
            "clean": "a", "noisy": "a"}
    rows = [{**base, "status": VALID, "cosine": 1.0, "euclidean": 0.0},
            {**base, "status": VALID, "cosine": 0.5, "euclidean": 2.0},
            {**base, "status": EMPTY, "cosine": None, "euclidean": None, "n_noisy_pieces": 0,
             "noisy": ""}]
    s = summarize_condition({"condition_id": "x", "operation": "deletion", "p": 0.2, "seed": 11}, rows)
    assert (s["selected"], s["valid"], s["undefined"], s["undefined_empty"]) == (3, 2, 1, 1)
    assert math.isclose(s["coverage"], 2 / 3)
    assert math.isclose(s["mean_cosine"], 0.75) and math.isclose(s["mean_euclidean"], 1.0)


# ---- the frozen table E -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def E():
    return load_E()[0]


def test_E_shape_checksum_and_read_only(E):
    assert E.shape == E_SHAPE and E.dtype == np.float32
    assert not E.flags.writeable
    with pytest.raises(ValueError):
        E[0, 0] = 1.0
    load_E()  # raises if the file no longer matches its recorded sha256


def test_pooling_is_the_plain_mean_of_raw_rows(E):
    pooled = pool_pieces(E, [5, 9, 5])
    assert np.allclose(pooled, (2 * E[5].astype(np.float64) + E[9]) / 3)
    assert pool_pieces(E, []) is None


def test_embedder_adds_no_special_tokens(E):
    from src.tokenizer.wrapper import TokenizerWrapper
    emb = SpanEmbedder(TokenizerWrapper().tokenizer, E)
    pieces, vec = emb.embed("lobster")
    assert pieces == ["lobster"] and vec.shape == (768,)  # no [CLS]/[SEP]
    assert emb.embed("")[1] is None
