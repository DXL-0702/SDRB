import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.train import (
    DataConfig,
    ExperimentConfig,
    GateConfig,
    ModelConfig,
    OptimizerConfig,
    OutputConfig,
    SchedulerConfig,
    TrainConfig,
    TrainingConfig,
    build_model,
    build_optimizer,
    build_scheduler,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


def _config(model_name: str) -> TrainConfig:
    gate = None
    stage_dilations = None
    if model_name == "sdrb_resnet20":
        gate = GateConfig(
            temperature_start=5.0,
            temperature_end=0.5,
            entropy_weight=0.01,
            metric_batches=1,
            image_interval=20,
        )
        stage_dilations = {
            "stage2": (1, 1, 2),
            "stage3": (1, 2, 3),
            "stage4": (1, 2, 5),
        }

    return TrainConfig(
        experiment=ExperimentConfig(name=f"smoke_{model_name}"),
        seed=42,
        device="cpu",
        data=DataConfig(
            dataset="cifar100",
            root="data",
            download=False,
            batch_size=2,
            num_workers=0,
        ),
        model=ModelConfig(name=model_name, num_classes=100, stage_dilations=stage_dilations),
        gate=gate,
        train=TrainingConfig(epochs=2, log_interval=1, eval_interval=1),
        optimizer=OptimizerConfig(name="sgd", lr=0.01, momentum=0.9, weight_decay=0.0005),
        scheduler=SchedulerConfig(name="cosine", eta_min=0.0),
        output=OutputConfig(runs_dir="runs", checkpoint_dir="checkpoints"),
    )


def _loader() -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    images = torch.randn(4, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3])
    dataset = TensorDataset(images, targets)
    return DataLoader(dataset, batch_size=2)


def test_baseline_one_train_step_loss_is_finite() -> None:
    config = _config("resnet20")
    model = build_model(config)
    optimizer = build_optimizer(config, model)
    criterion = nn.CrossEntropyLoss()

    metrics = train_one_epoch(
        model,
        _loader(),
        criterion,
        optimizer,
        torch.device("cpu"),
        epoch=1,
        epochs=config.train.epochs,
    )

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert torch.isfinite(torch.tensor(metrics["top1"]))


def test_sdrb_one_train_step_loss_and_entropy_are_finite() -> None:
    config = _config("sdrb_resnet20")
    model = build_model(config)
    optimizer = build_optimizer(config, model)
    criterion = nn.CrossEntropyLoss()

    metrics = train_one_epoch(
        model,
        _loader(),
        criterion,
        optimizer,
        torch.device("cpu"),
        epoch=1,
        epochs=config.train.epochs,
        gate_config=config.gate,
    )

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert torch.isfinite(torch.tensor(metrics["entropy_loss"]))
    assert "gate/stage2/entropy_norm" in metrics
    assert "gate/stage3/prob_branch0" in metrics
    assert "gate/stage4/argmax_frac_branch2" in metrics


def test_tiny_eval_loop_returns_loss_and_top1() -> None:
    config = _config("resnet20")
    model = build_model(config)
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(model, _loader(), criterion, torch.device("cpu"))

    assert set(metrics) == {"loss", "top1"}
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert torch.isfinite(torch.tensor(metrics["top1"]))


def test_checkpoint_save_and_load_uses_tmp_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config("resnet20")
    model = build_model(config)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    checkpoint_path = tmp_path / "last.pt"

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        epoch=1,
        best_top1=12.5,
        config=config,
        metrics={"eval": {"top1": 12.5}},
    )

    loaded_model = build_model(config)
    loaded_optimizer = build_optimizer(config, loaded_model)
    loaded_scheduler = build_scheduler(config, loaded_optimizer)
    checkpoint = load_checkpoint(
        checkpoint_path,
        loaded_model,
        loaded_optimizer,
        loaded_scheduler,
        torch.device("cpu"),
    )

    assert checkpoint["epoch"] == 1
    assert checkpoint["best_top1"] == 12.5
