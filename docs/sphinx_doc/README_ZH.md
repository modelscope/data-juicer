# Data-Juicer 文档

Data-Juicer 借助 Sphinx 构建 API 文档。
如需更新生成的文档，请运行以下命令：

```bash
# $~/data_juicer/docs/sphinx_doc
# 1.安装 sphinx 的依赖并初始化 sphinx-quickstart
pip install sphinx sphinx-autobuild sphinx_rtd_theme recommonmark
# or pip install -r ../../environments/dev_requires
# 2. 运行文档构建脚本
./build_doc.sh

# 3. 构建完成的文档存储目录为 `build/html`
mv build/html position_to_publish
```

- 为了方便起见（不必再次从头开始编译），可以按如下方式下载构建的目录（包括 html 文件）：

```bash
# cd docs/sphinx_doc 
wget https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/sphinx_API_build_0801.zip
unzip sphinx_API_build_0801.zip
```
