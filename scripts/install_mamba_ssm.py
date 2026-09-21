"""Install the prebuilt mamba-ssm wheel matching the active Torch/CUDA runtime."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request

import torch


def main() -> None:
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401

        print("mamba-ssm selective-scan kernel: PASS")
        return
    except Exception as error:
        print("Installing matching mamba-ssm wheel:", repr(error))

    version = "2.3.2.post1"
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_tag = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    if not torch.version.cuda:
        raise RuntimeError("Mamba-UNet requires a CUDA PyTorch runtime")
    cuda_tag = "cu12" if int(torch.version.cuda.split(".")[0]) >= 12 else "cu11"
    abi = str(bool(torch._C._GLIBCXX_USE_CXX11_ABI)).upper()
    wheel = (
        f"mamba_ssm-{version}+{cuda_tag}torch{torch_tag}cxx11abi{abi}-"
        f"{python_tag}-{python_tag}-linux_x86_64.whl"
    )
    api = f"https://api.github.com/repos/state-spaces/mamba/releases/tags/v{version}"
    with urllib.request.urlopen(api, timeout=30) as response:
        assets = json.load(response)["assets"]
    urls = {asset["name"]: asset["browser_download_url"] for asset in assets}
    if wheel not in urls:
        candidates = sorted(
            name for name in urls if python_tag in name and cuda_tag in name and f"torch{torch_tag}" in name
        )
        raise RuntimeError(f"No exact wheel for {wheel}. Runtime-compatible candidates: {candidates}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", urls[wheel]],
        check=True,
    )
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401

    print("Installed:", wheel)


if __name__ == "__main__":
    main()

