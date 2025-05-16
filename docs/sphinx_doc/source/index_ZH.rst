.. _Data Processing for and with Foundation Models:
.. role:: raw-html-m2r(raw)
   :format: html


`[英文主页] <../index>`_ | `[DJ-Cookbook] <DJ-cookbook_ZH>`_ | `[算子池] <docs/Operators>`_ | `[API] <../api>`_ | `[Awesome LLM Data] <docs/awesome_llm_data>`_

Data Processing for and with Foundation Models
==============================================

 :raw-html-m2r:`<img src="https://img.alicdn.com/imgextra/i1/O1CN01fUfM5A1vPclzPQ6VI_!!6000000006165-0-tps-1792-1024.jpg" width = "533" height = "300" alt="Data-Juicer"/>`


.. image:: https://img.shields.io/badge/language-Python-214870.svg
   :target: https://img.shields.io/badge/language-Python-214870.svg
   :alt: 


.. image:: https://img.shields.io/badge/license-Apache--2.0-000000.svg
   :target: https://img.shields.io/badge/license-Apache--2.0-000000.svg
   :alt: 


.. image:: https://img.shields.io/pypi/v/py-data-juicer?logo=pypi&color=026cad
   :target: https://pypi.org/project/py-data-juicer
   :alt: pypi version


.. image:: https://img.shields.io/docker/v/datajuicer/data-juicer?logo=docker&label=Docker&color=498bdf
   :target: https://hub.docker.com/r/datajuicer/data-juicer
   :alt: Docker version


.. image:: https://img.shields.io/badge/OSS%20latest-none?logo=docker&label=Docker&color=498bdf
   :target: https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/docker_images/data-juicer-latest.tar.gz
   :alt: Docker on OSS


.. image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FHYLcool%2Ff856b14416f08f73d05d32fd992a9c29%2Fraw%2Ftotal_cov.json
   :target: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FHYLcool%2Ff856b14416f08f73d05d32fd992a9c29%2Fraw%2Ftotal_cov.json
   :alt: 



.. image:: https://img.shields.io/badge/DataModality-Text,Image,Audio,Video-brightgreen.svg
   :target: ../DJ-cookbook
   :alt: DataModality


.. image:: https://img.shields.io/badge/Usage-Cleaning,Synthesis,Analysis-FFD21E.svg
   :target: ../DJ-cookbook
   :alt: Usage


.. image:: https://img.shields.io/badge/ModelScope-Demos-4e29ff.svg?logo=data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjI0IDEyMS4zMyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxwYXRoIGQ9Im0wIDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtOTkuMTQgNzMuNDloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xNzYuMDkgOTkuMTRoLTI1LjY1djIyLjE5aDQ3Ljg0di00Ny44NGgtMjIuMTl6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTEyNC43OSA0Ny44NGgyNS42NXYyNS42NWgtMjUuNjV6IiBmaWxsPSIjMzZjZmQxIiAvPgoJPHBhdGggZD0ibTAgMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xOTguMjggNDcuODRoMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xOTguMjggMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xNTAuNDQgMHYyMi4xOWgyNS42NXYyNS42NWgyMi4xOXYtNDcuODR6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTczLjQ5IDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiMzNmNmZDEiIC8+Cgk8cGF0aCBkPSJtNDcuODQgMjIuMTloMjUuNjV2LTIyLjE5aC00Ny44NHY0Ny44NGgyMi4xOXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtNDcuODQgNzMuNDloLTIyLjE5djQ3Ljg0aDQ3Ljg0di0yMi4xOWgtMjUuNjV6IiBmaWxsPSIjNjI0YWZmIiAvPgo8L3N2Zz4K
   :target: https://modelscope.cn/studios?name=Data-Jiucer&page=1&sort=latest&type=1
   :alt: ModelScope- Demos


.. image:: https://img.shields.io/badge/🤗HuggingFace-Demos-4e29ff.svg
   :target: https://huggingface.co/spaces?&search=datajuicer
   :alt: HuggingFace- Demos



.. image:: https://img.shields.io/badge/Doc-DJ_Cookbook-blue?logo=Markdown
   :target: ../DJ-cookbook
   :alt: Document_List


.. image:: https://img.shields.io/badge/文档-DJ指南-blue?logo=Markdown
   :target: DJ-cookbook_ZH
   :alt: 文档列表


.. image:: https://img.shields.io/badge/文档-算子池-blue?logo=Markdown
   :target: docs/Operators
   :alt: 算子池


.. image:: http://img.shields.io/badge/cs.LG-1.0Paper(SIGMOD'24)-B31B1B?logo=arxiv&logoColor=red
   :target: https://arxiv.org/abs/2309.02033
   :alt: Paper


.. image:: http://img.shields.io/badge/cs.AI-2.0Paper-B31B1B?logo=arxiv&logoColor=red
   :target: https://arxiv.org/abs/2501.14755
   :alt: Paper


Data-Juicer 是一个一站式系统，面向大模型的文本及多模态数据处理。我们提供了一个基于 JupyterLab 的 `Playground <http://8.138.149.181/>`_\ ，您可以从浏览器中在线试用 Data-Juicer。 如果Data-Juicer对您的研发有帮助，请支持加星（自动订阅我们的新发布）、以及引用我们的\ `工作 <#参考文献>`_ 。

`阿里云人工智能平台 PAI <https://www.aliyun.com/product/bigdata/learn>`_ 已引用Data-Juicer并将其能力集成到PAI的数据处理产品中。PAI提供包含数据集管理、算力管理、模型工具链、模型开发、模型训练、模型部署、AI资产管理在内的功能模块，为用户提供高性能、高稳定、企业级的大模型工程化能力。数据处理的使用文档请参考：\ `PAI-大模型数据处理 <https://help.aliyun.com/zh/pai/user-guide/components-related-to-data-processing-for-foundation-models/?spm=a2c4g.11186623.0.0.3e9821a69kWdvX>`_\ 。

Data-Juicer正在积极更新和维护中，我们将定期强化和新增更多的功能和数据菜谱。热烈欢迎您加入我们（issues/PRs/\ `Slack频道 <https://join.slack.com/t/data-juicer/shared_invite/zt-23zxltg9d-Z4d3EJuhZbCLGwtnLWWUDg?spm=a2c22.12281976.0.0.7a8275bc8g7ypp>`_ /\ `钉钉群 <https://qr.dingtalk.com/action/joingroup?code=v1,k1,YFIXM2leDEk7gJP5aMC95AfYT+Oo/EP/ihnaIEhMyJM=&_dt_no_comment=1&origin=11>`_\ /...），一起推进大模型的数据-模型协同开发和研究应用！

----

新消息
------
.. include:: README_ZH.md
    :start-after: ## 新消息
    :end-before: 目录
    :parser: myst_parser.sphinx_

目录
====


* `新消息 <#id7>`_
* `为什么选择 Data-Juicer？ <#id14>`_
.. toctree::
   :maxdepth: 2
   :caption: 教程

   DJ-cookbook_ZH
   Installation_ZH
   quick-start_ZH

* `开源协议 <#id15>`_
* `贡献 <#id16>`_
* `致谢 <#id17>`_
* `参考文献 <#id18>`_

为什么选择 Data-Juicer？
------------------------

.. include:: README_ZH.md
    :start-after: ## 为什么选择 Data-Juicer？
    :end-before: ## DJ-Cookbook
    :parser: myst_parser.sphinx_

开源协议
--------

.. include:: README_ZH.md
    :start-after: ## 开源协议
    :end-before: ## 贡献
    :parser: myst_parser.sphinx_

贡献
----

.. include:: README_ZH.md
    :start-after: ## 贡献
    :end-before: ## 致谢
    :parser: myst_parser.sphinx_

致谢
----

.. include:: README_ZH.md
    :start-after: ## 致谢
    :end-before: ## 参考文献
    :parser: myst_parser.sphinx_

参考文献
--------

.. include:: README_ZH.md
    :start-after: ## 参考文献
    :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 2
   :caption: 帮助文档
   :glob:
   :hidden:

   docs/*

.. toctree::
   :maxdepth: 2
   :caption: demos
   :glob:
   :hidden:

   demos/*
   demos/**/*

.. toctree::
   :maxdepth: 2
   :caption: 工具
   :glob:
   :hidden:

   tools/*
   tools/**/*

.. toctree::
   :maxdepth: 2
   :caption: 第三方
   :glob:
   :hidden:

   thirdparty/*
   thirdparty/**/*

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api