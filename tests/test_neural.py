import math

import numpy as np
import torch

from src.lm.vocab import Vocab
from src.neural.data import BOS, encode_pieces
from src.neural.evaluate import evaluate
from src.neural.model import NextTokenNet
from src.neural.train import pick_best_epoch, train_model


def _vocab():
    return Vocab(name="t", tokens=["</s>", "[UNK]", "a", "b"], unk="[UNK]")


BERT = {"a": 10, "b": 11, "[UNK]": 0}


def test_framing_matches_assignment_table():
    v = _vocab()
    s = encode_pieces([["a", "b"], ["a"], []], v, BERT)
    a, b, eos = v.index["a"], v.index["b"], v.end_id
    assert s.ctx.tolist() == [[BOS, BOS], [BOS, 10], [10, 11],      # record 1: -> a, b, EOS
                              [BOS, BOS], [BOS, 10],                # record 2 (one token): -> a, EOS
                              [BOS, BOS]]                           # record 3 (empty): -> EOS
    assert s.tgt.tolist() == [a, b, eos, a, eos, eos]
    assert s.T == 6 and s.n_records == 3 and s.n_empty_records == 1 and s.n_tokens == 3


def test_unknown_piece_targets_are_counted():
    v = _vocab()
    s = encode_pieces([["[UNK]", "a"]], v, BERT)
    assert s.n_unk_targets == 1 and s.tgt[0] == v.unk_id


def _net(mode, d=4, K=5, h=3, V=20, seed=0):
    torch.manual_seed(seed)
    return NextTokenNet(torch.randn(V, d), mode, K, h)


def test_bos_is_an_all_zero_vector():
    net = _net("mean")
    e = net.embed(torch.tensor([-1, 3]))
    assert torch.count_nonzero(e[0]) == 0 and torch.equal(e[1], net.E[3])


def test_parameter_counts_and_frozen_table():
    d, K, h = 4, 5, 3
    mean, cat = _net("mean"), _net("concat")
    assert mean.counts() == {"trainable": h * d + h + K * h + K, "frozen": 20 * d}
    assert cat.counts()["trainable"] == h * 2 * d + h + K * h + K
    assert all("E" not in n for n, _ in mean.named_parameters())      # E is not a parameter
    assert "E" not in mean.state_dict()                               # and is not saved with the weights


def test_swapping_context_tokens():
    ctx, swapped = torch.tensor([[2, 7]]), torch.tensor([[7, 2]])
    mean, cat = _net("mean"), _net("concat")
    assert torch.allclose(mean(ctx), mean(swapped), atol=1e-6)        # the mean cannot see the order
    assert not torch.allclose(cat(ctx), cat(swapped), atol=1e-4)      # concatenation can


def test_mean_always_divides_by_two_even_with_bos():
    net = _net("mean")
    x = net.features(torch.tensor([[-1, 5]]))
    assert torch.allclose(x[0], net.E[5] / 2)


def test_best_epoch_ties_keep_the_earlier_epoch():
    assert pick_best_epoch([3.0, 2.0, 2.0, 2.5]) == 2
    assert pick_best_epoch([2.0, 2.0, 2.0]) == 1
    assert pick_best_epoch([3.0, 2.5, 2.0]) == 3


def test_hand_computed_loss_perplexity_accuracy():
    net = _net("mean", K=3)
    with torch.no_grad():
        net.fc1.weight.zero_(); net.fc1.bias.zero_(); net.fc2.weight.zero_()
        net.fc2.bias.copy_(torch.log(torch.tensor([0.5, 0.25, 0.25])))
    ctx = torch.tensor([[1, 2]] * 4)
    tgt = torch.tensor([0, 1, 2, 0])
    r = evaluate(net, ctx, tgt, chunk=3)                 # chunking must not change the result
    S = -(2 * math.log(0.5) + 2 * math.log(0.25))
    assert abs(r["S"] - S) < 1e-5 and abs(r["H"] - S / 4) < 1e-6
    assert abs(r["PP"] - math.exp(S / 4)) < 1e-5 and r["accuracy"] == 0.5 and r["T"] == 4


def test_untrained_uniform_model_has_loss_ln_k():
    net = _net("concat", K=7)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    r = evaluate(net, torch.tensor([[1, 2]] * 5), torch.tensor([0, 1, 2, 3, 4]))
    assert abs(r["H"] - math.log(7)) < 1e-6 and abs(r["PP"] - 7) < 1e-4


def test_training_is_reproducible_and_leaves_E_unchanged():
    torch.manual_seed(1)
    E = torch.randn(30, 4)
    E0 = E.clone()
    ctx = torch.randint(-1, 30, (40, 2)); tgt = torch.randint(0, 6, (40,))
    cfg = {"seed": 101, "hidden_size": 3, "learning_rate": 0.01, "weight_decay": 0.0, "batch_size": 8,
           "max_epochs": 3, "eval_chunk": 16}
    a = train_model("mean", E, 6, (ctx, tgt), (ctx, tgt), cfg)
    b = train_model("mean", E, 6, (ctx, tgt), (ctx, tgt), cfg)
    assert torch.equal(E, E0)
    for (n1, p1), (n2, p2) in zip(a["model"].state_dict().items(), b["model"].state_dict().items()):
        assert torch.equal(p1, p2), n1
    assert [h["val_loss"] for h in a["history"]] == [h["val_loss"] for h in b["history"]]
    assert sum(h["selected"] for h in a["history"]) == 1
