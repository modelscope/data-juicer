#!/bin/bash
sphinx-apidoc -f -o source ../../data_juicer -t _templates
python ./create_symlinks.py
make clean
languages=("zh-CN")

make -e SPHINXOPTS="-D language=en" html

for lang in "${languages[@]}"; do
    make -e SPHINXOPTS="-D language=$lang" -e BUILDDIR="build/$lang" html
    cp -r build/$lang/html build/html/$lang
done

