from __future__ import annotations

from templatestudio.multimodal import (
    count_media_parts,
    image_part,
    text_part,
    user_message,
    validate_content_parts,
)


def test_valid_content_parts_pass():
    content = [text_part("what is this?"), image_part("http://x/y.jpg")]
    assert validate_content_parts(content) == []


def test_unknown_part_type_flagged():
    problems = validate_content_parts([{"type": "hologram", "data": 1}])
    assert any("unknown type" in p for p in problems)


def test_missing_required_field_flagged():
    problems = validate_content_parts([{"type": "image"}])
    assert any("missing 'url'" in p for p in problems)


def test_count_media_parts_counts_only_media():
    convo = [
        user_message([text_part("a"), image_part("u1"), image_part("u2")]),
        {"role": "assistant", "content": "ok"},
    ]
    assert count_media_parts(convo) == 2


def test_user_message_shape():
    msg = user_message([text_part("hi")])
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)


# --- two-stage rendering: what an image costs after expansion ---

import pytest  # noqa: E402

from templatestudio.multimodal import (  # noqa: E402
    GEMMA3_IMAGE_TOKENS,
    count_expanded_placeholders,
    qwen_vl_image_tokens,
)

QWEN3_VL_BOUNDS = {"patch_size": 16, "min_pixels": 65536, "max_pixels": 16777216}


def test_qwen_vl_448_square():
    assert qwen_vl_image_tokens(448, 448) == 256                    # Qwen2/2.5-VL
    assert qwen_vl_image_tokens(448, 448, **QWEN3_VL_BOUNDS) == 196  # Qwen3-VL


def test_qwen_vl_large_photo_is_downscaled_or_not_by_the_bounds():
    assert qwen_vl_image_tokens(3000, 4000) == 1230
    assert qwen_vl_image_tokens(3000, 4000, **QWEN3_VL_BOUNDS) == 11750


def test_qwen_vl_full_hd_frame():
    assert qwen_vl_image_tokens(1080, 1920) == 1222
    assert qwen_vl_image_tokens(1080, 1920, **QWEN3_VL_BOUNDS) == 2040


def test_gemma3_cost_is_constant():
    assert GEMMA3_IMAGE_TOKENS == 256


def test_count_expanded_placeholders():
    assert count_expanded_placeholders([1, 9, 9, 9, 2, 9], 9) == 4


@pytest.mark.parametrize("size", [(448, 448), (1080, 1920), (32, 32), (500, 333)])
def test_matches_the_real_qwen2_vl_processor(size):
    pytest.importorskip("PIL")
    vl = pytest.importorskip("transformers.models.qwen2_vl.image_processing_qwen2_vl")
    from PIL import Image

    h, w = size
    processor = vl.Qwen2VLImageProcessor(patch_size=14, merge_size=2)
    grid = processor(images=Image.new("RGB", (w, h)), return_tensors="np")[
        "image_grid_thw"
    ][0]
    assert int(grid.prod()) // 4 == qwen_vl_image_tokens(h, w)
