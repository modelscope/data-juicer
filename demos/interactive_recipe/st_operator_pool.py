import streamlit as st
from loguru import logger
import emoji
from collections import OrderedDict
import numpy as np
import pandas as pd
import os
import re
import json
import math

from matplotlib import pyplot as plt
from wordcloud import WordCloud

from data_juicer.utils.constant import Fields, StatsKeys

from operator_pool import OperatorArg, Operator, OperatorPool


class StOperatorArg(OperatorArg):

    def _on_v_change(self):
        new_v = st.session_state.get(f"{self.op.name}_{self.name}")
        try:
            self.set_v(new_v)
            if self.stats_apply and self.quantiles is not None:
                st.session_state[f"{self.op.name}_{self.name}_p"] = self._v2p(self.v)
        except Exception as e:
            logger.error(e)
            st.session_state[f"{self.op.name}_{self.name}"] = self.v

    def _on_p_change(self):
        new_p = st.session_state.get(f"{self.op.name}_{self.name}_p")
        try:
            self.set_p(new_p)
            st.session_state[f"{self.op.name}_{self.name}"] = self.v
        except Exception as e:
            logger.error(e)
            st.session_state[f"{self.op.name}_{self.name}_p"] = self._v2p(self.v)

    def render(self):
        if self.type == 'bool':
            st.selectbox(
                self.name,
                options=self.v_options,
                # index=0 if self.v else 1,
                key=f"{self.op.name}_{self.name}",
                help=self.desc,
                on_change=self._on_v_change,
            )
        elif self.type in ['int', 'float']:
            step = 1 if self.v_type == int else 0.01
            if self.stats_apply and self.quantiles is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input(
                        self.name,
                        min_value=self.v_min,
                        max_value=self.v_max,
                        # value=self.v,
                        step=step,
                        key=f"{self.op.name}_{self.name}",
                        help=self.desc,
                        on_change=lambda: self._on_v_change()
                    )
                with col2:
                    st.number_input(
                        "quantile",
                        min_value=0,
                        max_value=100,
                        # value=self._v2p(self.v),
                        step=1,
                        key=f"{self.op.name}_{self.name}_p",
                        on_change=lambda: self._on_p_change(),
                    )
            else:
                st.number_input(
                    self.name,
                    min_value=self.v_min,
                    max_value=self.v_max,
                    step=step,
                    key=f"{self.op.name}_{self.name}",
                    help=self.desc,
                    on_change=lambda: self._on_v_change()
                )
        elif self.type == 'str':
            if self.v_options is not None:
                st.selectbox(
                    self.name,
                    options=self.v_options,
                    key=f"{self.op.name}_{self.name}",
                    help=self.desc,
                    on_change=self._on_v_change,
                )
            else:
                st.text_input(
                    label=self.name,
                    key=f"{self.op.name}_{self.name}",
                    help=self.desc,
                    on_change=lambda: self._on_v_change(),
                )
        elif self.type == 'list_str':
            st.multiselect(
                label=self.name,
                key=f"{self.op.name}_{self.name}",
                options=self.v_options,
                help=self.desc,
                on_change=lambda: self._on_v_change(),
            )
        else:
            raise NotImplementedError

    def st_sync(self):
        st.session_state[f'{self.op.name}_{self.name}'] = self.v
        if self.stats_apply is not None and self.quantiles is not None:
            st.session_state[f'{self.op.name}_{self.name}_p'] = self._v2p(self.v)


class StOperator(Operator):
    def __init__(self, pool, state: dict = None):
        super(StOperator, self).__init__(pool, state)
        for arg_name in self.args:
            self.args[arg_name] = StOperatorArg(self, self.args[arg_name].state)

    @property
    def require_quantile_plot(self):
        return self.dj_stats_key is not None

    def render(self):
        show_enabled_only = st.session_state.get("show_enabled_only")
        if show_enabled_only and not self.enabled:
            # Don't render op if show_enabled_only option is on and op is not enabled
            return
        # Op window start
        op_enabled_icon = ":check_mark_button:" if self.enabled else ":cross_mark:"
        with st.expander(emoji.emojize(f"{op_enabled_icon} :hammer_and_wrench:{self.name}"), expanded=False):
            # Short description of op
            st.text(self.desc)
            # op enable button
            st.checkbox(
                'enabled',
                key=f"{self.name}_enabled",
                on_change=self.disable if self.enabled else self.enable
            )
            # render args
            if self.enabled:
                for arg_name in self.args:
                    self.args[arg_name].render()
                # Quantile plot in the end
                if self.quantiles is not None:
                    chart_data = pd.DataFrame(
                        np.array(self.quantiles).reshape(-1, 1),
                        columns=['quantile'],
                    )
                    st.line_chart(chart_data)

            if self.name == "language_id_score_filter":
                # display word cloud of language id
                if st.session_state.get("analyzed_dataset", None) is not None:
                    stats = st.session_state.analyzed_dataset[Fields.stats]
                    language_ids = [s[StatsKeys.lang] for s in stats]
                    fig, ax = plt.subplots()
                    text = ' '.join(language_ids)
                    wordcloud = WordCloud(width=800, height=400, background_color='white', random_state=0).generate(text)
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

    def st_sync(self):
        st.session_state[f'{self.name}_enabled'] = self.enabled
        for arg_name in self.args:
            self.args[arg_name].st_sync()


class StOperatorPool(OperatorPool):
    def __init__(self, config_path=None):
        super(StOperatorPool, self).__init__(config_path=config_path)
        for op_name in self.pool:
            self.pool[op_name] = StOperator(self, state=self.pool[op_name].state)
        self.items_per_page = 15  # Number of operators displayed per page
        self.current_page = 1
        self.search_term = ""
        self.search_example = "e.g., filter|remove"

    def filter_operators(self):
        if not self.search_term:
            return list(self.pool.keys())

        filtered_ops = []
        try:
            pattern = re.compile(self.search_term.lower())
            for op_name, op in self.pool.items():
                if pattern.search(op_name.lower()) or pattern.search(op.desc.lower()):
                    filtered_ops.append(op_name)
        except re.error:
            st.warning("Invalid regular expression.")
            return list(self.pool.keys())

        return filtered_ops

    def render(self):
        with st.sidebar:
            st.header(emoji.emojize(":toolbox:Operator Pool"))
            # Export config
            btn_export_cfg = st.button("Export Config", use_container_width=True)
            if btn_export_cfg:
                config_path = self.export_config(
                    project_name=st.session_state.project_name,
                    dataset_path=st.session_state.dataset_path,
                    nproc=1,
                    export_path=st.session_state.get(
                        "export_path", "./data/processed_dataset.jsonl"
                    ),
                )
                config_path = os.path.join(os.path.dirname(__file__), config_path)
                st.write(f"Successfully export config to {config_path}.")

            # search box
            self.search_term = st.text_input(
                "Search Operators",
                value=self.search_term,
                placeholder=self.search_example,
            )
            # Show enabled only option
            st.checkbox(
                emoji.emojize("Show enabled:check_mark_button: only"),
                value=False,
                key="show_enabled_only",
            )

            # filter operator
            filtered_ops = self.filter_operators()

            total_ops = len(filtered_ops)

            # Computes the operator displayed on the current page
            start_index = (self.current_page - 1) * self.items_per_page
            end_index = min(start_index + self.items_per_page, total_ops)
            ops_to_render = filtered_ops[start_index:end_index]

            # Operator rendering the current page
            for op_name in ops_to_render:
                self.pool[op_name].render()

            total_pages = math.ceil(total_ops / self.items_per_page)

            # paging controls
            cols = st.columns([0.3, 0.4, 0.3])
            with cols[0]:
                if st.button("👈", disabled=self.current_page == 1):
                    self.current_page -= 1
            with cols[2]:
                if st.button(
                    "👉", disabled=self.current_page == total_pages or total_pages == 0
                ):
                    self.current_page += 1

            st.write(f"Page {self.current_page} of {total_pages}")

    def st_sync(self):
        for op_name in self.pool:
            self.pool[op_name].st_sync()
