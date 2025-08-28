# remove_bibliography_mapper

Removes bibliography sections at the end of LaTeX documents.

This operator identifies and removes bibliography sections in LaTeX documents. It uses a
regular expression to match common bibliography commands such as ppendix,
egin{references}, egin{thebibliography}, and ibliography. The matched sections are
removed from the text. The operator processes samples in batch mode for efficiency.

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_bibliography_case

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">%%
%% This is file `sample-sigconf.tex\clearpage
\bibliographystyle{ACM-Reference-Format}
\bibliography{sample-base}
\end{document}
\endinput
%%
%% End of file `sample-sigconf.tex&#x27;.
</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">%%
%% This is file `sample-sigconf.tex\clearpage
\bibliographystyle{ACM-Reference-Format}
</pre></div>

#### ✨ explanation 解释
The operator removes the bibliography section, including the \bibliographystyle and \bibliography commands, from the end of the LaTeX document, leaving only the initial part of the text up to where the bibliography starts.
算子从LaTeX文档末尾移除参考文献部分，包括\bibliographystyle和\bibliography命令，只保留从文档开始到参考文献开始前的部分文本。

### test_ref_case

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">%%
%% This is file `sample-sigconf.tex\clearpage
\begin{references}
\end{document}
\endinput
%%
%% End of file `sample-sigconf.tex&#x27;.
</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">%%
%% This is file `sample-sigconf.tex\clearpage
</pre></div>

#### ✨ explanation 解释
The operator identifies and removes the bibliography section starting with \begin{references} until the end of the document, resulting in the removal of everything after \clearpage.
算子识别并移除从\begin{references}开始直到文档末尾的参考文献部分，导致\clearpage之后的所有内容都被删除。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/remove_bibliography_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_remove_bibliography_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)