# Data-Juicer Documentation

We build our API documentation with help of Sphinx.
To update the generated
doc, please run the following commands:

```bash
# $~/data_juicer/docs/sphinx_doc
# 1. install the sphinx requirements and init the sphinx-quickstart
pip install sphinx sphinx-autobuild sphinx_rtd_theme recommonmark
# or pip install -r ../../environments/dev_requires

# 2. auto generate and build the doc
./build_doc.sh

# 3. finalize the doc, which is stored in the `build/html` directory
mv build/html position_to_publish
```

Automatic action in github can be found in [here](https://github.com/modelscope/data-juicer/blob/main/.github/workflows/deploy_sphinx_docs.yml).