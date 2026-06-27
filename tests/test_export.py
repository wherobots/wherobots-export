from pathlib import Path

import pytest
import torch
from torch.export.pt2_archive._package import load_pt2

from wherobots_export.torch.export import create_example_input_from_shape, save


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.conv(pixels).mean(dim=(2, 3))


class Normalize(torch.nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return (pixels - 0.5) / 0.5


def test_save_creates_nonempty_pt2(tmp_path: Path) -> None:
    output_file = tmp_path / "model.pt2"
    save(model=TinyModel(), output_file=output_file, input_shape=[1, 3, 32, 32], device="cpu")
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_save_dynamic_batch_roundtrip(tmp_path: Path) -> None:
    model = TinyModel().eval()
    output_file = tmp_path / "model.pt2"
    save(model=model, output_file=output_file, input_shape=[-1, 3, 32, 32], device="cpu")

    contents = load_pt2(str(output_file))
    assert set(contents.exported_programs) == {"model"}

    exported = contents.exported_programs["model"].module()
    pixels = torch.randn(4, 3, 32, 32)  # batch differs from the export-time example
    torch.testing.assert_close(exported(pixels), model(pixels))


def test_save_with_transforms_packages_both(tmp_path: Path) -> None:
    output_file = tmp_path / "model.pt2"
    save(
        model=TinyModel(),
        output_file=output_file,
        input_shape=[-1, 3, 32, 32],
        device="cpu",
        transforms=Normalize(),
    )
    contents = load_pt2(str(output_file))
    assert set(contents.exported_programs) == {"model", "transforms"}


def test_create_example_input_rejects_all_dynamic() -> None:
    with pytest.raises(ValueError):
        create_example_input_from_shape([-1, -1, -1, -1])


def test_create_example_input_fills_dynamic_dims() -> None:
    example = create_example_input_from_shape([-1, 3, -1, -1], shape_default_value=64)
    assert example.shape == (2, 3, 64, 64)
