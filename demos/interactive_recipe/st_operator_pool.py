import streamlit as st
from loguru import logger
import emoji
import numpy as np
import pandas as pd
import os
import re
import math
import yaml
import copy

from matplotlib import pyplot as plt
from wordcloud import WordCloud

from data_juicer.utils.constant import Fields, StatsKeys

from operator_pool import OperatorArg, Operator, OperatorPool
from recipe_utils import RecipeManager


all_ops_config_path = os.path.join(os.path.dirname(__file__), "./configs/all_ops.yaml") or os.path.join(os.path.dirname(__file__), "./configs/default_ops.yaml")

with open(all_ops_config_path, "r") as f:
    all_ops = yaml.safe_load(f)

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
    def __init__(self, config_path=None, default_ops=None):
        super(StOperatorPool, self).__init__(config_path=config_path, default_ops=default_ops)
        for op_name in self.pool:
            self.pool[op_name] = StOperator(self, state=self.pool[op_name].state)
        self.items_per_page = 8  # Number of operators displayed per page
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        if 'search_term' not in st.session_state:
            st.session_state.search_term = ""
        
        recipes_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../../configs/data_juicer_recipes'
        ))
        self.recipe_manager = RecipeManager(recipes_path, all_ops)
            
        self.current_page = st.session_state.current_page
        self.search_example = "e.g., filter|language"

    def filter_operators(self, search_term, op_list):
        if not search_term:
            return op_list

        try:
            pattern = re.compile(search_term.lower())
            return [op_name for op_name in op_list if pattern.search(op_name.lower())]
        except re.error:
            st.warning("Invalid regular expression.")
            return op_list

    def _cleanup_edit_dialog_state(self):
        """Helper function to clean up session state after dialog closes."""
        keys_to_delete = ['edited_op_pool_names', 'edit_op_search_term']
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
    
    def add_ops_and_clear_callback(self, ops_to_add):
        st.session_state.edited_op_pool_names.update(ops_to_add)
        st.session_state.edit_op_search_term = ""

    @st.dialog("Edit Operator Pool", width="large")
    def render_edit_op_pool_dialog(self):
        # 1. Initialize temporary state on first open
        if 'edited_op_pool_names' not in st.session_state:
            st.session_state.edited_op_pool_names = set(self.pool.keys())
        if 'edit_op_current_page' not in st.session_state:
            st.session_state.edit_op_current_page = 1

        st.info(
            "This demo is still under development, so the available operators are currently incomplete."
            "We provide over 100 operators. For the complete list of operators and their descriptions, please refer to "
            "[this page](https://modelscope.github.io/data-juicer/en/main/docs/Operators.html)."
        )
        col_active, col_available = st.columns(2, gap="large")

        # ==================================================================
        # LEFT COLUMN: Active Operators
        # ==================================================================
        with col_active:
            # Use columns to place header and "Remove All" button on the same line
            header_col, btn_col = st.columns([0.6, 0.4])
            with header_col:
                st.subheader("Active Operators")
            
            active_ops_list = sorted(list(st.session_state.edited_op_pool_names))
            
            with btn_col:
                # Disable button if there are no active operators
                if st.button("Remove All", key="remove_all_ops", disabled=not active_ops_list, use_container_width=True):
                    st.session_state.edited_op_pool_names.clear()
                    st.rerun()

            if not active_ops_list:
                st.caption("No operators in the pool. Add some from the right!")
            
            with st.container(height=420): # Adjusted height slightly
                for op_name in active_ops_list:
                    row_col1, row_col2 = st.columns([0.8, 0.2])
                    with row_col1:
                        st.markdown(f"**{op_name}**")
                    with row_col2:
                        st.button(
                            "➖", 
                            key=f"remove_op_{op_name}", 
                            help=f"Remove {op_name}",
                            on_click=st.session_state.edited_op_pool_names.remove,
                            args=(op_name,)
                            )
        # ==================================================================
        # RIGHT COLUMN: Available Operators
        # ==================================================================
        with col_available:
            header_col, btn_col = st.columns([0.5, 0.5])
            with header_col:
                st.subheader("Available Operators")
            
            available_ops_list = sorted(list(set(all_ops.keys()) - st.session_state.edited_op_pool_names))

            search_term = st.text_input(
                "Search available operators",
                key="edit_op_search_term",
                placeholder=self.search_example,
            )
            
            filtered_available_ops = self.filter_operators(search_term, available_ops_list)

            with btn_col:
                # Disable button if the filtered list is empty
                st.button(
                    "Add All Searched", 
                    key="add_all_ops", 
                    disabled=not filtered_available_ops, 
                    use_container_width=True,
                    on_click=self.add_ops_and_clear_callback,
                    args=(filtered_available_ops,)
                    )

            ops_to_display = filtered_available_ops

            with st.container(height=350): # Adjusted height
                if not ops_to_display:
                    st.caption("No matching operators found.")
                for op_name in ops_to_display:
                    row_col1, row_col2 = st.columns([0.8, 0.2])
                    with row_col1:
                        st.write(op_name)
                    with row_col2:
                        st.button("➕", key=f"add_op_{op_name}", help=f"Add {op_name}", on_click=self.add_ops_and_clear_callback, args=([op_name],))

        # ==================================================================
        # DIALOG ACTIONS: Apply or Cancel
        # ==================================================================
        st.markdown("---")
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            if st.button("Apply Changes", type="primary", use_container_width=True):
                original_ops = set(self.pool.keys())
                edited_ops = st.session_state.edited_op_pool_names
                
                ops_to_remove = original_ops - edited_ops
                ops_to_add = edited_ops - original_ops
                
                for op in ops_to_remove:
                    logger.info(f"Removing operator: {op}")
                    del self.pool[op]
                
                for op in ops_to_add:
                    logger.info(f"Adding operator: {op}")
                    self.add_op(op, all_ops[op])
                
                self.st_sync() # IMPORTANT: Sync changes back to widget states
                self._cleanup_edit_dialog_state()
                st.session_state.current_page = 1
                st.rerun()

        with btn_col2:
            if st.button("Cancel", use_container_width=True):
                self._cleanup_edit_dialog_state()
                st.rerun()

    @st.dialog("Reuse Example Recipe")
    def render_reuse_example_recipe_dialog(self):
        st.info(
            "Select a pre-configured recipe to start with. "
            "This will **replace** your current operator pool."
            "For more information, please refer to [RecipeGallery](https://modelscope.github.io/data-juicer/en/main/docs/RecipeGallery.html)."
            )

        # Search bar for recipes
        search_term = st.text_input("Search recipes by name", key="recipe_search_term").lower()

        # Filter recipes based on search term
        if search_term:
            filtered_recipes = [r for r in self.recipe_manager.recipes if search_term in r.name.lower()]
        else:
            filtered_recipes = self.recipe_manager.recipes

        if not self.recipe_manager.recipes:
             st.error(f"No recipes found in the directory: {self.recipe_manager.recipes_dir}. Please check the path.")
             if st.button("Close"):
                st.rerun()
             return
        elif not filtered_recipes:
            st.warning("No recipes match your search.")

        # Display recipes as a list of expandable cards
        with st.container(height=500):
            for recipe in filtered_recipes:
                with st.expander(f"**{recipe.name}**"):
                    st.markdown("---") # Visual separator
                    
                    # Iterate through each operator and its configuration in the recipe
                    for op_name, op_config in recipe.operators.items():
                        st.markdown(f"**- {op_name}**")
                        
                        # Check if there are arguments to display
                        args_from_recipe = op_config.get('args', {})
                        if args_from_recipe:
                            for arg_name, arg_details in args_from_recipe.items():
                                param_col1, param_col2 = st.columns([0.4, 0.6])

                                with param_col1:
                                    st.caption(f"`{arg_name}`")

                                with param_col2:
                                    v = arg_details.get('v')
                                    if v is not None:
                                        display_v = v
                                    else:
                                        display_v = arg_details.get("default")

                                    if isinstance(display_v, list):
                                        display_v = f"[{', '.join(map(str, display_v))}]"
                                    st.caption(f": {display_v}")
                        else:
                            st.caption("  *(no parameters)*")
                        
                        st.text("") # Add a little vertical space between operators

                    st.markdown("---")
                    if st.button("Apply this Recipe", key=f"apply_{recipe.path}", type="primary", use_container_width=True):
                        logger.info(f"Applying recipe: {recipe.name}")
                        
                        # 1. Clear the current operator pool
                        self.pool.clear()
                        
                        # 2. Add all operators from the selected recipe
                        for op_name, op_config_to_add in recipe.operators.items():
                            self.add_op(op_name, op_config_to_add)
                            for arg_name, arg_details in op_config_to_add.get('args', {}).items():
                                if arg_details.get("v") is not None:
                                    self.act(op_name=op_name, action_type="set_arg",
                                                  arg_name=arg_name, v=arg_details["v"])
                        
                        # 3. Sync state, close dialog, and refresh the app
                        self.st_sync()
                        st.session_state.current_page = 1
                        st.rerun()

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

            # Button to open the new dialog
            if st.button("⚙️ Edit Operator Pool", use_container_width=True):
                self.render_edit_op_pool_dialog()
                
            if st.button("🍽️ Reuse Example Recipe", use_container_width=True):
                self.render_reuse_example_recipe_dialog()

            # Show enabled only option
            st.checkbox(
                emoji.emojize("Show enabled:check_mark_button: only"),
                value=False,
                key="show_enabled_only",
            )
            
            self.current_page = st.session_state.current_page

            total_ops = list(self.pool.keys())
            total_pages = math.ceil(len(total_ops) / self.items_per_page) if len(total_ops) > 0 else 1

            # Computes the operator displayed on the current page
            start_index = (self.current_page - 1) * self.items_per_page
            end_index = min(start_index + self.items_per_page, len(total_ops))
            ops_to_render = total_ops[start_index:end_index]

            # Operator rendering the current page
            for op_name in ops_to_render:
                self.pool[op_name].render()

            # paging controls
            cols = st.columns([0.3, 0.4, 0.3])
            with cols[0]:
                if st.button("←", disabled=self.current_page <= 1):
                    st.session_state.current_page -= 1
                    st.rerun()

            with cols[2]:
                if st.button(
                    "→", disabled=self.current_page >= total_pages
                ):
                    st.session_state.current_page += 1
                    st.rerun()

            st.write(f"Page {self.current_page} of {total_pages}")
            
    def add_op(self, op_name, arg_state):
        state_copy = copy.deepcopy(arg_state)
        state_copy.update(name=op_name)
        self.pool[op_name] = StOperator(self, state=state_copy)

    def st_sync(self):
        for op_name in self.pool:
            self.pool[op_name].st_sync()

