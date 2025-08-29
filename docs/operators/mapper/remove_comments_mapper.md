# remove_comments_mapper

Removes comments from documents, currently supporting only 'tex' format.

This operator removes inline and multiline comments from text samples. It supports both
inline and multiline comment removal, controlled by the `inline` and `multiline`
parameters. Currently, it is designed to work with 'tex' documents. The operator
processes each sample in the batch and applies regular expressions to remove comments.
The processed text is then updated in the original samples.

- Inline comments are removed using the pattern `[^\]%.+$`.
- Multiline comments are removed using the pattern `^%.*
?`.

Important notes:
- Only 'tex' document type is supported at present.
- The operator processes the text in place and updates the original samples.

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `doc_type` | typing.Union[str, typing.List[str]] | `'tex'` | Type of document to remove comments. |
| `inline` | <class 'bool'> | `True` | Whether to remove inline comments. |
| `multiline` | <class 'bool'> | `True` | Whether to remove multiline comments. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_tex_case
```python
RemoveCommentsMapper(doc_type='tex', inline=True, multiline=True)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">%%
%% This is file `sample-sigconf.tex&#x27;,
%% The first command in your LaTeX source must be the \documentclass command.
\documentclass[sigconf,review,anonymous]{acmart}
%% NOTE that a single column version is required for 
%% submission and peer review. This can be done by changing
\input{math_commands.tex}
%% end of the preamble, start of the body of the document source.
\begin{document}
%% The &quot;title&quot; command has an optional parameter,
\title{Hierarchical Cross Contrastive Learning of Visual Representations}
%%
%% The &quot;author&quot; command and its associated commands are used to define
%% the authors and their affiliations.
\author{Hesen Chen}
\affiliation{%
  \institution{Alibaba Group}
  \city{Beijing}
  \country{China}}
\email{hesen.chs@alibaba-inc.com}
%% By default, the full list of authors will be used in the page
\begin{abstract}The rapid
\end{abstract}
\begin{CCSXML}
\ccsdesc[500]{Computing methodologies~Image representations}
%% Keywords. The author(s) should pick words that accurately describe
\keywords{self-supervised,  ontrastive Learning, hierarchical projection, cross-level}
%% page.
\begin{teaserfigure}
\end{teaserfigure}
%% This command processes the author and affiliation and title
\maketitle
\section{Introduction}
\begin{itemize}
\end{itemize}
\section{Related Work}
\label{gen_inst} Self-supervised
\section{Method}
\label{method}In this section,
\subsection{Framework} kkk
\subsection{Cross Contrastive Loss}
Since $\sZ^n$ are extracted
\subsection{Implementation details}
\textbf{Image augmentations} We use
\textbf{Architecture} We use
\textbf{Optimization} We adapt 
\section{Experiments}
\label{experiments}In this section
\subsection{Linear and Semi-Supervised Evaluations on ImageNet}
\textbf{Linear evaluation on ImageNet} We firs
\textbf{Semi-supervised learning on ImageNet} We simply
\subsection{Transfer to other datasets and tasks}
\textbf{Image classification with fixed features} We follow
\section{Ablations} We present
\subsection{Influence of hierarchical projection head and cross contrastive loss} get out
\subsection{Levels and depth of projector network}
\end{center}
\caption{\label{figure3} \textbf{Different way of cross-correlation on 3 level hierarchical projection head.} &#x27;=&#x27; denotes stop gradient.}
\end{figure}
\subsection{Analyze of} In this
\textbf{Similarity between} Using SimSiam
\textbf{Feature similarity} We extracted
\section{Conclusion}
We propose HCCL
\clearpage
\bibliographystyle{ACM-Reference-Format}
\bibliography{sample-base}
\end{document}
\endinput
%%
%% End of file `sample-sigconf.tex&#x27;.
</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">\documentclass[sigconf,review,anonymous]{acmart}
\input{math_commands.tex}
\begin{document}
\title{Hierarchical Cross Contrastive Learning of Visual Representations}
\author{Hesen Chen}
\affiliation{%
  \institution{Alibaba Group}
  \city{Beijing}
  \country{China}}
\email{hesen.chs@alibaba-inc.com}
\begin{abstract}The rapid
\end{abstract}
\begin{CCSXML}
\ccsdesc[500]{Computing methodologies~Image representations}
\keywords{self-supervised,  ontrastive Learning, hierarchical projection, cross-level}
\begin{teaserfigure}
\end{teaserfigure}
\maketitle
\section{Introduction}
\begin{itemize}
\end{itemize}
\section{Related Work}
\label{gen_inst} Self-supervised
\section{Method}
\label{method}In this section,
\subsection{Framework} kkk
\subsection{Cross Contrastive Loss}
Since $\sZ^n$ are extracted
\subsection{Implementation details}
\textbf{Image augmentations} We use
\textbf{Architecture} We use
\textbf{Optimization} We adapt 
\section{Experiments}
\label{experiments}In this section
\subsection{Linear and Semi-Supervised Evaluations on ImageNet}
\textbf{Linear evaluation on ImageNet} We firs
\textbf{Semi-supervised learning on ImageNet} We simply
\subsection{Transfer to other datasets and tasks}
\textbf{Image classification with fixed features} We follow
\section{Ablations} We present
\subsection{Influence of hierarchical projection head and cross contrastive loss} get out
\subsection{Levels and depth of projector network}
\end{center}
\caption{\label{figure3} \textbf{Different way of cross-correlation on 3 level hierarchical projection head.} &#x27;=&#x27; denotes stop gradient.}
\end{figure}
\subsection{Analyze of} In this
\textbf{Similarity between} Using SimSiam
\textbf{Feature similarity} We extracted
\section{Conclusion}
We propose HCCL
\clearpage
\bibliographystyle{ACM-Reference-Format}
\bibliography{sample-base}
\end{document}
\endinput
</pre></div>

#### ✨ explanation 解释
The RemoveCommentsMapper operator is used to remove both inline and multiline comments from a TeX document. In this example, the operator is configured to remove both types of comments. The input text contains various comments, such as lines starting with '%%' and lines with '%' in the middle. After processing, all these comments are removed, leaving only the essential parts of the document, like LaTeX commands and content. This results in a cleaner, more readable version of the document without any of the original comments.
RemoveCommentsMapper算子用于从TeX文档中移除行内和多行注释。在这个例子中，算子被配置为移除两种类型的注释。输入文本包含各种注释，比如以'%%'开头的行以及中间有'%'的行。处理后，所有这些注释都被移除，只保留了文档中的关键部分，如LaTeX命令和内容。这使得文档更干净、更易读，且不包含任何原始注释。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/remove_comments_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_remove_comments_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)