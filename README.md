# 🔭 `mininear` [![Stars](https://img.shields.io/github/stars/althonos/mininear.svg?style=social&maxAge=3600&label=Star)](https://github.com/althonos/mininear/stargazers)

*A [NumPy](https://numpy.org/) port of the [`NEAR`](https://github.com/TravisWheelerLab/NEAR) code for embedding protein sequences.*

[![Actions](https://img.shields.io/github/actions/workflow/status/althonos/mininear/test.yml?branch=main&logo=github&style=flat-square&maxAge=300)](https://github.com/althonos/mininear/actions)
[![Coverage](https://img.shields.io/codecov/c/gh/althonos/mininear?style=flat-square&maxAge=3600)](https://codecov.io/gh/althonos/mininear/)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/bsd-3-clause/)
[![PyPI](https://img.shields.io/pypi/v/mininear.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/mininear)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/mininear?style=flat-square&maxAge=3600&logo=anaconda)](https://anaconda.org/bioconda/mininear)
[![Wheel](https://img.shields.io/pypi/wheel/mininear.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/mininear/#files)
[![Python Versions](https://img.shields.io/pypi/pyversions/mininear.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/mininear/#files)
[![Python Implementations](https://img.shields.io/badge/impl-universal-success.svg?style=flat-square&maxAge=3600&label=impl)](https://pypi.org/project/mininear/#files)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/mininear/)
[![Mirror](https://img.shields.io/badge/mirror-LUMC-003EAA.svg?maxAge=2678400&style=flat-square)](https://git.lumc.nl/mflarralde/mininear)
[![GitHub issues](https://img.shields.io/github/issues/althonos/mininear.svg?style=flat-square&maxAge=600)](https://github.com/althonos/mininear/issues)
[![Docs](https://img.shields.io/readthedocs/mininear/latest?style=flat-square&maxAge=600)](https://mininear.readthedocs.io)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/mininear/blob/master/CHANGELOG.md)
[![Downloads](https://img.shields.io/pypi/dm/mininear?style=flat-square&color=303f9f&maxAge=86400&label=downloads)](https://pepy.tech/project/mininear)

## 🗺️ Overview

[`NEAR`](https://github.com/TravisWheelerLab/NEAR) (*neural embeddings for amino acid relationships*) 
is a method developed by Daniel Olson *et al.*[\[1\]](#ref1) to generate
meaningfull residue-level embeddings for proteins, which can then be used
for nearest-neighbor search in high-dimensional space.

`mininear` is a pure-Python package to encode protein sequences into NEAR 
embeddings, using the trained weights from the NEAR ResNet, and a portable
re-implementation of ResNet using NumPy (and code derived from the 
[`numpy-ml`](https://github.com/ddbourgin/numpy-ml) project).

This library only depends on NumPy and is available for all modern Python
versions (3.7+).

<!-- ### 📋 Features -->


## 🔧 Installing

Install the `mininear` package from GitHub, until a PyPI release is made 
available:
```console
$ pip install git+https://github.com/althonos/mininear
```

<!-- Install the `mininear` package directly from [PyPi](https://pypi.org/project/mininear)
which hosts universal wheels that can be installed with `pip`:
```console
$ pip install mininear
``` -->

<!-- Otherwise, `mininear` is also available as a [Bioconda](https://bioconda.github.io/)
package:
```console
$ conda install -c bioconda mininear
``` -->

<!-- ## 📖 Documentation

A complete [API reference](https://mininear.readthedocs.io/en/stable/api.html)
can be found in the [online documentation](https://mininear.readthedocs.io/),
or directly from the command line using
[`pydoc`](https://docs.python.org/3/library/pydoc.html):
```console
$ pydoc mininear
``` -->

## 💡 Example

`mininear` provides a single `Encoder` class, which can be used to encode
a sequence using the NEAR ResNet:
```python
import mininear

encoder = mininear.Encoder()
embedding = encoder.encode_sequence("MLELLPTAVEGVSQAQITGRPEWIWLALGTALMGLGTL...")
```

The embedding for that sequence is returned as a NumPy array. Currently
there is no batching support, but this is planned if that is a desirable
feature.


## 💭 Feedback

### ⚠️ Issue Tracker

Found a bug? Have an enhancement request? Head over to the [GitHub issue
tracker](https://github.com/althonos/mininear/issues) if you need to report
or ask something. If you are filing in on a bug, please include as much
information as you can about the issue, and try to recreate the same bug
in a simple, easily reproducible situation.

### 🏗️ Contributing

Contributions are more than welcome! See
[`CONTRIBUTING.md`](https://github.com/althonos/mininear/blob/main/CONTRIBUTING.md)
for more details.

## 📋 Changelog

This project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html)
and provides a [changelog](https://github.com/althonos/mininear/blob/master/CHANGELOG.md)
in the [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) format.

## ⚖️ License

This library is provided under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/). 
It includes some code adapted from the [`numpy-ml`](https://github.com/ddbourgin/numpy-ml)
which is also released under the GPL 3.0. The NEAR code and weights are 
released under the [BSD-3-Clause License](https://choosealicense.com/licenses/bsd-3-clause/)
and are redistributed and adapted here under those terms.

*This project is in no way not affiliated, sponsored, or otherwise endorsed
by the [original `NEAR` authors](https://github.com/TravisWheelerLab).
It was developed by [Martin Larralde](https://github.com/althonos/) during his
PhD project at the [Leiden University Medical Center](https://www.lumc.nl/)
in the [Zeller team](https://github.com/zellerlab).*


## 📚 References

- <a id="ref1">\[1\]</a> Daniel Olson, Thomas Colligan, Daphne Demekas, Jack W Roddy, Ken Youens-Clark, Travis J Wheeler, NEAR: neural embeddings for amino acid relationships, *Bioinformatics*, Volume 41, Issue Supplement_1, July 2025, Pages i449–i457, [doi:10.1093/bioinformatics/btaf198](https://doi.org/10.1093/bioinformatics/btaf198).
