import torch

from models import build_cifar_model, cifar_resnet20, cifar_sdrb_resnet20


def test_cifar_resnet20_forward_shape() -> None:
    model = cifar_resnet20(num_classes=100)
    x = torch.randn(2, 3, 32, 32)

    logits = model(x)

    assert logits.shape == (2, 100)


def test_cifar_sdrb_resnet20_forward_shape() -> None:
    model = cifar_sdrb_resnet20(num_classes=100)
    x = torch.randn(2, 3, 32, 32)

    logits = model(x)

    assert logits.shape == (2, 100)


def test_cifar_sdrb_resnet20_returns_stagewise_gates() -> None:
    model = cifar_sdrb_resnet20(num_classes=100)
    x = torch.randn(2, 3, 32, 32)

    logits, gates_by_stage = model(x, return_gates=True, temperature=1.0)

    assert logits.shape == (2, 100)
    assert set(gates_by_stage) == {"stage2", "stage3", "stage4"}
    assert len(gates_by_stage["stage2"]) == 3
    assert len(gates_by_stage["stage3"]) == 3
    assert len(gates_by_stage["stage4"]) == 3

    for gates in gates_by_stage.values():
        for gate in gates:
            assert gate.shape[0] == 2
            assert gate.shape[1] == 3
            torch.testing.assert_close(gate.sum(dim=1), torch.ones_like(gate[:, 0]))


def test_build_cifar_model_builds_resnet20() -> None:
    model = build_cifar_model("resnet20", num_classes=100)
    x = torch.randn(2, 3, 32, 32)

    logits = model(x)

    assert logits.shape == (2, 100)


def test_build_cifar_model_builds_sdrb_resnet20() -> None:
    model = build_cifar_model("sdrb_resnet20", num_classes=100)
    x = torch.randn(2, 3, 32, 32)

    logits = model(x)

    assert logits.shape == (2, 100)
