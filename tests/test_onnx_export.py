import io
from typing import Any, cast

import pytest
import torch
from ops.sdrb_block import SDRBBlock


def test_sdrb_block_onnx_export_smoke() -> None:
    onnx = pytest.importorskip("onnx")
    block = SDRBBlock(in_channels=8, out_channels=8, stride=1, use_gumbel=False)
    block.eval()
    x = torch.randn(1, 8, 16, 16)
    buffer = io.BytesIO()

    torch.onnx.export(
        block,
        (x,),
        cast(Any, buffer),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )

    model = onnx.load_from_string(buffer.getvalue())
    onnx.checker.check_model(model)
