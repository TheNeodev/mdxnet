from setuptools import setup, find_packages

setup(
    name="mdxnet",
    version="1.0.0",
    author="Your Name",
    author_email="",
    description="Ultimate Vocal Remover using MDX Net",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/theneodev/mdxnet",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "librosa",
        "soundfile",
        "torch",
        "onnxruntime",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
