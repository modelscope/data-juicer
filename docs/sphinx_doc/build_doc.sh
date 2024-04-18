#!/bin/bash
sphinx-apidoc -f -o source ../../data_juicer -t _templates
make clean html