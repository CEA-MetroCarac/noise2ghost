# Noise2Ghost

[![ci](https://github.com/CEA-MetroCarac/noise2ghost/workflows/ci/badge.svg)](https://github.com/CEA-MetroCarac/noise2ghost/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://CEA-MetroCarac.github.io/noise2ghost/)
[![pypi version](https://img.shields.io/pypi/v/noise2ghost.svg)](https://pypi.org/project/noise2ghost/)
[![gitter](https://badges.gitter.im/join%20chat.svg)](https://gitter.im/noise2ghost/community)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Self-supervised deep convolutional reconstructions for ghost imaging.

## Getting Started

It takes a few steps to setup Noise2Ghost on your machine. We recommend installing
[Anaconda package manager](https://www.anaconda.com/download/) or the more compact
[Miniconda](https://docs.anaconda.com/free/miniconda/index.html) for Python 3.

### Installing with conda

Simply install with:
```bash
conda install noise2ghost -c n-vigano
```

### Installing from PyPI

Simply install with:
```bash
python3 -m pip install noise2ghost
```

If you are on jupyter, and don't have the rights to install packages system-wide, then you can install with:
```python
! pip install --user noise2ghost
```

### Installing from source

To install Noise2Ghost, simply clone this github.com project with either:
```bash
git clone https://github.com/CEA-MetroCarac/noise2ghost.git noise2ghost
```
or:
```bash
git clone git@github.com:CEA-MetroCarac/noise2ghost.git noise2ghost
```

Then go to the cloned directory and run `pip` installer:
```bash
cd noise2ghost
pip install -e .
```

## How to contribute

Contributions are always welcome. Please submit pull requests against the `main` branch.

If you have any issues, questions, or remarks, then please open an issue on github.com.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.