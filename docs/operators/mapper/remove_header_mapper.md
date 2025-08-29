# remove_header_mapper

Removes headers at the beginning of documents in LaTeX samples.

This operator identifies and removes headers such as chapter, part, section, subsection,
subsubsection, paragraph, and subparagraph. It uses a regular expression to match these
headers. If a sample does not contain any headers and `drop_no_head` is set to True, the
sample text will be removed. Otherwise, the sample remains unchanged. The operator
processes samples in batches for efficiency.

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `drop_no_head` | <class 'bool'> | `True` | whether to drop sample texts without |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_case

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
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">\section{Introduction}
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

#### ✨ explanation 解释
The operator removes the header and preamble sections from a LaTeX document, leaving only the main content. In this example, all the initial parts of the document such as the document class, title, author, and abstract are removed, and the text starting from the 'Introduction' section is kept. This is useful for processing documents where the focus is on the main body rather than the metadata or formatting commands.
算子从LaTeX文档中移除标题和前言部分，只保留主要内容。在这个例子中，文档的初始部分如文档类、标题、作者和摘要都被移除了，而从'Introduction'章节开始的文本被保留下来。这对于处理关注主体内容而非元数据或格式命令的文档非常有用。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/remove_header_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_remove_header_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)