# MDXNet
Ultimate Vocal Remover using MDX Net.

## Features
- High-quality vocal separation using MDX Net.
- GPU acceleration (if available).
- Supports multi-threaded processing for speed optimization.

## Installation
```sh
pip install git+https://github.com/TheNeodev/mdxnet.git
```

## Usage
## Command Line

```
mdxnet-run --input input_audio.wav --model path/to/model.onnx --output output_directory

```

## Python API


```
from mdxnet.mdx_runner import MDXRunner

model_params = {
    # Add your model parameter dictionary here
}

runner = MDXRunner(
    model_params=model_params,
    output_dir="output",
    model_path="path/to/model.onnx",
    filename="input_audio.wav",
    exclude_main=False,
    exclude_inversion=False,
    denoise=False,
    keep_orig=True,
    m_threads=2
)
main_fp, invert_fp = runner.run()

print("Processed files:", main_fp, invert_fp)
```
