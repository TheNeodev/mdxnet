# MDXNet  
**Ultimate Vocal Remover powered by MDX Net**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://python.org)
[![GitHub](https://img.shields.io/badge/GitHub-TheNeodev%2Fmdxnet-blue.svg)](https://github.com/TheNeodev/mdxnet)

MDXNet is a high-quality vocal separation tool that uses the MDX Net architecture. It leverages GPU acceleration (when available) and multi-threaded processing to deliver fast and efficient separation of vocals from audio files.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Downloading Models](#downloading-models)
- [Usage](#usage)
  - [Python API](#python-api)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- **High-Quality Vocal Separation:** Utilizes MDX Net for precise separation.
- **GPU Acceleration:** Automatically uses GPU if available.
- **Multi-Threaded Processing:** Optimized for faster processing on multi-core systems.

---

## Installation

Install MDXNet directly from GitHub using pip:

```sh
pip install git+https://github.com/TheNeodev/mdxnet.git
```

Make sure you have Python 3.7 or later installed.

---

## Downloading Models

MDXNet requires pre-trained models to operate. Download the required models from the releases page:

- **Model Repository:**  
  [Download Models](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/)

After downloading, place the model file (e.g., `uvr_models.onnx`) in a directory of your choice, and update the model path accordingly in your configuration.

---

## Usage

MDXNet can be used within your Python scripts.


### Python API

Below is an example of how to use the Python API for vocal separation:

```python
from mdxnet import MDXProcessor

# Define your model parameters
model_params = {
    # Customize model parameters here
    # e.g., "param1": value, "param2": value,
}

if __name__ == "__main__":
    # Initialize the processor with the model path and parameters.
    processor = MDXProcessor(
        model_path="./uvr_models.onnx",  # Update this path to your downloaded model
        model_params=model_params,
        processor=0  # Set processor index (use 0 for CPU, or specify GPU device index)
    )

    # Process the audio file to separate vocals and instrumental tracks.
    main_path, invert_path = processor.process(
        input_path="./Test.mp3",   # Path to the input audio file
        output_dir="./output",     # Output directory for the separated tracks
        denoise=True,              # Enable denoising (set to False if not needed)
        suffix="Vocals",           # Suffix for the vocal track file
        invert_suffix="Instrumental"  # Suffix for the instrumental track file
    )

    print(f"Separated vocals saved to: {main_path}")
    print(f"Instrumental track saved to: {invert_path}")
```

**Notes:**
- Ensure the model file is correctly placed and the path is updated.
- Customize the `model_params` dictionary based on your specific requirements.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear messages.
4. Submit a pull request detailing your changes.

For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- Thanks to the developers behind MDX Net/UVR for their groundbreaking work.
- Special thanks to all contributors and the community for continuous support.

---

