"""The released checkpoints carry a two-layer post-smoothing in the gated-residual branch (App. A.3).

Loading such a checkpoint has to rebuild that module and apply it the way it was trained; switching
`gr_post_smooth` off afterwards gives the refinement without smoothing (the setting of Table 7).
Run with: python -m pytest tests/
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mast3r.model import AsymmetricMASt3R  # noqa: E402

inf = float("inf")
HEADS = ("downstream_head1", "downstream_head2")
POST = "downstream_head1.residual_gated."  # pred1 comes from head 1


def tiny_model(head_type="catmlp+dpt+xyz_niw_dpt+residual_gated"):
    torch.manual_seed(0)
    return AsymmetricMASt3R(
        pos_embed="RoPE100", patch_embed_cls="PatchEmbedDust3R", img_size=(32, 32),
        head_type=head_type, output_mode="pts3d+desc24",
        depth_mode=("exp", -inf, inf), conf_mode=("exp", 1, inf),
        enc_embed_dim=64, enc_depth=2, enc_num_heads=4, dec_embed_dim=64, dec_depth=10, dec_num_heads=4,
        two_confs=True, desc_conf_mode=("exp", 0, inf), landscape_only=False).eval()


def released_layout_state_dict(net):
    """Weights laid out like the released checkpoints, with a residual branch that is not a no-op."""
    g = torch.Generator().manual_seed(1)
    sd = {k: v.clone() for k, v in net.state_dict().items() if ".residual_gated." not in k}
    for head in HEADS:
        for name in ("gate_mlp.fc2.weight", "delta_mlp.fc2.weight"):  # zero in a fresh model
            sd[f"{head}.{name}"] = 0.05 * torch.randn(sd[f"{head}.{name}"].shape, generator=g)
        p = f"{head}.residual_gated."
        sd[p + "gate_post.weight"] = torch.randn(1, 1, 3, 3, generator=g)
        sd[p + "gate_post.bias"] = torch.randn(1, generator=g)
        sd[p + "delta_post.0.weight"] = torch.randn(3, 1, 3, 3, generator=g)
        sd[p + "delta_post.0.bias"] = torch.randn(3, generator=g)
        sd[p + "delta_post.1.weight"] = torch.randn(3, 3, 1, 1, generator=g)
        sd[p + "delta_post.1.bias"] = torch.randn(3, generator=g)
    return sd


def predict(net):
    g = torch.Generator().manual_seed(2)
    views = [dict(img=torch.randn(1, 3, 32, 32, generator=g), true_shape=torch.tensor([[32, 32]]), instance=[str(i)])
             for i in range(2)]
    with torch.no_grad():
        pred1, _ = net(*views)
    return pred1


def set_post_smooth(net, enabled):
    for head in HEADS:
        getattr(net, head).gr_post_smooth = enabled


@pytest.fixture(scope="module")
def loaded():
    net = tiny_model()
    sd = released_layout_state_dict(net)
    result = net.load_state_dict(sd, strict=False)
    with_smoothing = predict(net)
    set_post_smooth(net, False)
    without_smoothing = predict(net)
    set_post_smooth(net, True)
    return sd, result, with_smoothing, without_smoothing


def test_released_layout_loads_without_leftover_keys(loaded):
    _, result, _, _ = loaded
    assert list(result.missing_keys) == []
    assert list(result.unexpected_keys) == []


def test_released_layout_still_loads_into_a_model_without_gated_residual():
    sd = released_layout_state_dict(tiny_model())
    plain = tiny_model(head_type="catmlp+dpt")  # e.g. scoring the MASt3R confidence of a Trust3R checkpoint
    result = plain.load_state_dict(sd, strict=False)
    assert list(result.missing_keys) == []


def test_loaded_post_smoothing_is_applied_directly_to_gate_logits_and_residual(loaded):
    sd, _, with_smoothing, without_smoothing = loaded
    logits, delta = without_smoothing["gr_gate_logits"], without_smoothing["gr_delta"]
    gate = torch.sigmoid(F.conv2d(logits, sd[POST + "gate_post.weight"], sd[POST + "gate_post.bias"], padding=1))
    delta = F.conv2d(delta, sd[POST + "delta_post.0.weight"], sd[POST + "delta_post.0.bias"], padding=1, groups=3)
    delta = F.conv2d(delta, sd[POST + "delta_post.1.weight"], sd[POST + "delta_post.1.bias"])
    expected = with_smoothing["pts3d_raw"] + (gate * delta).permute(0, 2, 3, 1)
    assert torch.allclose(with_smoothing["pts3d"], expected, atol=1e-5)


def test_switching_post_smoothing_off_leaves_the_plain_gated_residual(loaded):
    _, _, with_smoothing, without_smoothing = loaded
    gate = torch.sigmoid(without_smoothing["gr_gate_logits"])
    expected = without_smoothing["pts3d_raw"] + (gate * without_smoothing["gr_delta"]).permute(0, 2, 3, 1)
    assert torch.allclose(without_smoothing["pts3d"], expected, atol=1e-5)
    assert not torch.allclose(without_smoothing["pts3d"], with_smoothing["pts3d"], atol=1e-3)
