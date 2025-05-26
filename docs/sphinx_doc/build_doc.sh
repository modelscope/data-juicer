#!/bin/bash
make clean
languages=(en zh_CN)

for lang in "${languages[@]}"; do
    sphinx-multiversion source build/$lang -D "language=$lang"
done

