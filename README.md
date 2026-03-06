# wherobots-export

Utilities for exporting PyTorch models and transform programs into `.pt2` archives used by Wherobots model onboarding workflows.

## Install

```bash
pip install wherobots-export
```

## Usage

```python
from pathlib import Path

import torch
from wherobots_export.torch.export import save

model = torch.nn.Identity()
save(
    model=model,
    output_file=Path("model.pt2"),
    input_shape=[1, 3, 224, 224],
    device="cpu",
)
```

## Release

This repo is configured for GitHub Actions publishing to PyPI using trusted publishing.

1. Create the package on PyPI (`wherobots-export`) and configure trusted publisher for this repo.
2. Push a tag like `v0.1.0`.
3. The `publish` workflow builds and uploads the release.
