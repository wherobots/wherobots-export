import inspect
import logging
from pathlib import Path

import torch
from torch.export.dynamic_shapes import Dim
from torch.export.pt2_archive._package import package_pt2

_logger = logging.getLogger(__name__)


def create_example_input_from_shape(
    input_shape: list[int], shape_default_value: int = 224
) -> torch.Tensor:
    shape = []

    if all(dim == -1 for dim in input_shape):
        raise ValueError(
            "Input shape cannot be all dynamic (-1). At least one dimension must be fixed."
        )
    elif any(dim != -1 for dim in input_shape):
        batch_dim = 2 if input_shape[0] == -1 else input_shape[0]
        shape.append(batch_dim)
        shape.extend([dim if dim != -1 else shape_default_value for dim in input_shape[1:]])
    else:
        shape = list(input_shape)

    return torch.randn(*shape, requires_grad=False)


def extract_module_arg_names(module: torch.nn.Module) -> str:
    if not hasattr(module, "forward"):
        raise ValueError("The provided module does not have a forward method.")

    return next(iter(inspect.signature(module.forward).parameters))


@torch.no_grad()
def _export(
    input_shape: list[int],
    model: torch.nn.Module,
    transforms: torch.nn.Module | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.export.ExportedProgram, torch.export.ExportedProgram | None]:
    example_inputs = create_example_input_from_shape(input_shape).to(device).to(dtype)
    dims = tuple(Dim.AUTO if dim == -1 else dim for dim in input_shape)
    _logger.info("Exporting with dims: %s", dims)
    _logger.info("Example input shape: %s", list(example_inputs.shape))

    model.eval()
    model = model.to(device).to(dtype)
    model_arg = extract_module_arg_names(model)
    model_program = torch.export.export(
        mod=model, args=(example_inputs,), dynamic_shapes={model_arg: dims}
    )

    if transforms is not None:
        transforms.eval()
        transforms = transforms.to(device).to(dtype)
        transforms_arg = extract_module_arg_names(transforms)
        transforms_program = torch.export.export(
            mod=transforms, args=(example_inputs,), dynamic_shapes={transforms_arg: dims}
        )
    else:
        transforms_program = None

    return model_program, transforms_program


def package(
    output_file: Path,
    model_program: torch.export.ExportedProgram,
    transforms_program: torch.export.ExportedProgram | None = None,
) -> None:
    exported_programs = {"model": model_program}
    if transforms_program is not None:
        exported_programs["transforms"] = transforms_program
    package_pt2(
        f=output_file,
        exported_programs=exported_programs,
        aoti_files=None,
    )


def save(
    model: torch.nn.Module,
    output_file: Path,
    input_shape: list[int],
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
    transforms: torch.nn.Module | None = None,
) -> None:
    """Saves a model and its transforms to a file using PyTorch's export mechanism.

    Parameters
    ----------
    model : torch.nn.Module
        The model to save.
    output_file : Path
        The path to the output file.
    input_shape : list[int]
        The shape of the input tensor, where -1 indicates a dynamic dimension.
    device : torch.device | None, optional
        The device to use for exporting. Default is CPU.
    dtype : torch.dtype, optional
        The data type to use for exporting. Default is torch.float32.
    transforms : torch.nn.Module | None, optional
        The transforms model to save. Default is None.
    """
    model_program, transforms_program = _export(
        model=model,
        transforms=transforms,
        input_shape=input_shape,
        device=device,
        dtype=dtype,
    )
    package(
        output_file=output_file,
        model_program=model_program,
        transforms_program=transforms_program,
    )
