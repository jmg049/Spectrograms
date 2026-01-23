#!/bin/bash
# Script to build and prepare docs for manual deployment

set -e

echo "Building documentation..."
cd docs
make clean
make html

echo "Documentation built successfully!"

cd build/html
zip -r ../../audio_samples_docs.zip .
cd ../..

echo "Created audio_samples_docs.zip ready for upload"