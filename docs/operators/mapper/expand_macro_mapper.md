# expand_macro_mapper

Expands macro definitions in the document body of LaTeX samples.

This operator processes LaTeX documents to expand user-defined macros in the text. It
supports \newcommand and \def macros without arguments. Macros are identified and
expanded in the text, ensuring they are not part of longer alphanumeric words. The
operator currently does not support macros with arguments. The processed text is updated
in the samples.

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_case

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">\documentclass{article}
% Recommended, but optional, packages for figures and better typesetting:
\usepackage{microtype}
\usepackage{graphicx}

% Attempt to make hyperref and algorithmic work together better:
\newcommand{\theHalgorithm}{\arabic{algorithm}}
% For theorems and such
\usepackage{amsmath}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THEOREMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\theoremstyle{plain}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}

\usepackage[textsize=small]{todonotes}
\setuptodonotes{inline}

\usepackage{makecell}
\newcommand{\cmark}{\ding{51}\xspace}%
\newcommand{\xmark}{\ding{55}\xspace}%

\def \alambic {\includegraphics[height=1.52ex]{img/alembic-crop.pdf}\xspace}

\newcommand\binke[1]{{\color{blue} \footnote{\color{blue}binke: #1}} }
\newcommand\Zerocost{Zero-cost}
\newcommand\imagenet{ImageNet}

\begin{document}

\begin{abstract}
The wide
\end{abstract}
\section{Introduction}
\label{introduction}
The main contributions are summarized as follows:
\section{Background and Related Work}\label{background}
\subsection{One-Shot NAS} In one-shot NAS
\section{PreNAS}\label{method}In this
\subsection{One-Shot NAS with Preferred Learning}
In the specialization stage, the optimal architectures under given  resource constraints can be directly obtained:
\begin{equation}
\widetilde{\mathcal{A}}^* = \widetilde{\mathcal{A}} .
\end{equation}
\subsection{Zero-Cost Transformer Selector}\label{sub:layerNorm}
\subsection{Performance Balancing} We discuss
\section{Experiments}\label{experiments}
\subsection{Setup}
\subsection{Main Results}\label{sec:sota}
\subsection{Analysis and Ablation study}\label{ablation}
\begin{figure}[t]
\vskip 0.1in
    \centering
    \subfigure[Search spaces]{\includegraphics[width=0.36\linewidth]{img/search_space.pdf}\label{fg:search_space:a}}%
    \hfil%
    \subfigure[Error distributions]{\includegraphics[width=0.58\linewidth]{img/cumulation.pdf}\label{fg:search_space:b}}
    \caption{Model quality}
\vskip -0.1in
\end{figure}
\paragraph{Effect of Performance Balancing} During
\subsection{Transfer Learning Results}
\subsection{CNN Results} in terms of similar FLOPs.
\FloatBarrier
\section{Conclusion}\label{conclusion} In this
% Acknowledgements should only appear in the accepted version.
\bibliography{ref}
\bibliographystyle{icml2023}
\clearpage
\appendix
\onecolumn
\section{Statistical}
\label{appendix:snipAnalysis} We analyze
\section{The Greedy Algorithm}
\label{appendix:greedy}
\section{Regularization \&amp; Data Augmentation}\label{appendix:aug}
\renewcommand{\arraystretch}{1.2}
\end{document}
</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">\documentclass{article}
% Recommended, but optional, packages for figures and better typesetting:
\usepackage{microtype}
\usepackage{graphicx}

% Attempt to make hyperref and algorithmic work together better:
\newcommand{\arabic{algorithm}}{\arabic{algorithm}}
% For theorems and such
\usepackage{amsmath}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THEOREMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\theoremstyle{plain}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}

\usepackage[textsize=small]{todonotes}
\setuptodonotes{inline}

\usepackage{makecell}
\newcommand{\cmark}{\ding{51}\xspace}%
\newcommand{\xmark}{\ding{55}\xspace}%

\def \includegraphics[height=1.52ex]{img/alembic-crop.pdf}\xspace {\includegraphics[height=1.52ex]{img/alembic-crop.pdf}\xspace}

\newcommand\binke[1]{{\color{blue} \footnote{\color{blue}binke: #1}} }
\newcommand\Zerocost{Zero-cost}
\newcommand\imagenet{ImageNet}

\begin{document}

\begin{abstract}
The wide
\end{abstract}
\section{Introduction}
\label{introduction}
The main contributions are summarized as follows:
\section{Background and Related Work}\label{background}
\subsection{One-Shot NAS} In one-shot NAS
\section{PreNAS}\label{method}In this
\subsection{One-Shot NAS with Preferred Learning}
In the specialization stage, the optimal architectures under given  resource constraints can be directly obtained:
\begin{equation}
\widetilde{\mathcal{A}}^* = \widetilde{\mathcal{A}} .
\end{equation}
\subsection{Zero-Cost Transformer Selector}\label{sub:layerNorm}
\subsection{Performance Balancing} We discuss
\section{Experiments}\label{experiments}
\subsection{Setup}
\subsection{Main Results}\label{sec:sota}
\subsection{Analysis and Ablation study}\label{ablation}
\begin{figure}[t]
\vskip 0.1in
    \centering
    \subfigure[Search spaces]{\includegraphics[width=0.36\linewidth]{img/search_space.pdf}\label{fg:search_space:a}}%
    \hfil%
    \subfigure[Error distributions]{\includegraphics[width=0.58\linewidth]{img/cumulation.pdf}\label{fg:search_space:b}}
    \caption{Model quality}
\vskip -0.1in
\end{figure}
\paragraph{Effect of Performance Balancing} During
\subsection{Transfer Learning Results}
\subsection{CNN Results} in terms of similar FLOPs.
\FloatBarrier
\section{Conclusion}\label{conclusion} In this
% Acknowledgements should only appear in the accepted version.
\bibliography{ref}
\bibliographystyle{icml2023}
\clearpage
\appendix
\onecolumn
\section{Statistical}
\label{appendix:snipAnalysis} We analyze
\section{The Greedy Algorithm}
\label{appendix:greedy}
\section{Regularization \&amp; Data Augmentation}\label{appendix:aug}
\renewcommand{\arraystretch}{1.2}
\end{document}
</pre></div>

#### ✨ explanation 解释
The operator expands the user-defined macros in the LaTeX document, replacing each macro with its defined content. In this case, \theHalgorithm is replaced with \arabic{algorithm} and \alambic is replaced with \includegraphics[height=1.52ex]{img/alembic-crop.pdf}\xspace. The rest of the document remains unchanged.
算子在LaTeX文档中扩展用户定义的宏，将每个宏替换为其定义的内容。在这个例子中，\theHalgorithm被替换为\arabic{algorithm}，而\alambic被替换为\includegraphics[height=1.52ex]{img/alembic-crop.pdf}\xspace。文档的其余部分保持不变。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/expand_macro_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_expand_macro_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)