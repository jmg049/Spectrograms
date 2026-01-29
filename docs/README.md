# Spectrograms Python Documentation

This directory contains the Sphinx documentation for the spectrograms Python library.

## Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main landing page
│   ├── api/                 # API reference documentation
│   │   ├── index.rst        # API overview
│   │   ├── parameters.rst   # Parameter classes
│   │   ├── functions.rst    # Convenience functions
│   │   ├── planner.rst      # Planner API
│   │   ├── result.rst       # Spectrogram result class
│   │   └── exceptions.rst   # Exception types
│   └── guide/               # User guide
│       ├── installation.rst
│       ├── quickstart.rst
│       ├── choosing_parameters.rst
│       ├── frequency_scales.rst
│       ├── planner_guide.rst
│       └── audio_features.rst
├── build/                   # Generated documentation (gitignored)
└── Makefile                 # Build commands
```

## Building Locally

### Quick build:

```bash
python ../build_docs.py
```

### With preview server:

```bash
python ../build_docs.py --serve
```

### Manual build:

```bash
make html
```

Then open `build/html/index.html` in your browser.

### Clean build:

```bash
python ../build_docs.py --clean
```

or

```bash
make clean
make html
```

## Documentation Style

- **Concise**: Keep explanations short and to the point
- **Educational**: Expand on complex topics with examples
- **Code examples**: Include working code snippets
- **Cross-references**: Link between related sections

## Autodoc

The documentation uses Sphinx autodoc to generate API reference from Python docstrings and type stubs. The type stubs in `python/spectrograms/__init__.pyi` provide the source for parameter and return type information.

## Extensions Used

- `sphinx.ext.autodoc`: Automatic API documentation from docstrings
- `sphinx.ext.autosummary`: Generate summary tables
- `sphinx.ext.napoleon`: Support for Google/NumPy style docstrings
- `sphinx.ext.viewcode`: Add links to highlighted source code

## Optional: Copy Button

To enable copy buttons on code blocks, install `sphinx-copybutton` and add it to `extensions` in `conf.py`:

```bash
pip install sphinx-copybutton
```

## Deployment

Use the deployment script to create a zip file for upload:

```bash
../deploy_docs.sh
```

