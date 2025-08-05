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
import json
import ast

from matplotlib import pyplot as plt
from wordcloud import WordCloud

from data_juicer.utils.constant import Fields, StatsKeys

from operator_pool import OperatorArg, Operator, OperatorPool
from recipe_utils import RecipeManager


all_ops_config_path = os.path.join(os.path.dirname(__file__), "./configs/all_op_info.yaml") or os.path.join(
    os.path.dirname(__file__), "./configs/default_ops.yaml"
)

with open(all_ops_config_path, "r") as f:
    all_ops = yaml.safe_load(f)


class StOperatorArg(OperatorArg):

    _TYPE_HANDLERS = {
        "list_types": frozenset(
            [
                "List[str]",
                "Optional[List[str]]",
                "List[int]",
                "Optional[List[int]]",
                "List[float]",
                "Optional[List[float]]",
                "List",
                "Union[str, List[str]]",
            ]
        ),
        "numeric_types": frozenset(["int", "float", "Optional[int]", "Optional[float]"]),
        "string_types": frozenset(["str", "Optional[str]"]),
        "bool_types": frozenset(["bool"]),
        "dict_types": frozenset(["Dict", "Optional[Dict]"]),
        "tuple_types": frozenset(["Tuple", "Optional[Tuple[int, int, int, int]]"]),
        "complex_types": frozenset(
            [
                "Union[str, int, None]",
                "Union[int, str]",
                "Union[str, int]",
                "Union[int, Tuple[int], Tuple[int, int], None]",
            ]
        ),
    }

    _COMPLEX_TYPE_CONFIGS = {
        "Union[str, List[str]]": {
            "placeholder": "Single value or multiple values (one per line)",
            "help": "Enter a single value or multiple values (one per line)",
            "component": "text_area",
        },
        "Union[str, int, None]": {
            "placeholder": "Text or number",
            "help": "Can be text or number",
            "component": "text_input",
        },
        "Union[int, str]": {
            "placeholder": "Text or number",
            "help": "Can be text or number",
            "component": "text_input",
        },
        "Union[str, int]": {
            "placeholder": "Text or number",
            "help": "Can be text or number",
            "component": "text_input",
        },
        "Union[int, Tuple[int], Tuple[int, int], None]": {
            "placeholder": "Examples: 5 | (5,) | (5, 10) | None",
            "help": "Examples:\n- Single number: 5\n- Tuple (1 value): (5,)\n- Tuple (2 values): (5, 10)\n- None: None or empty",
            "component": "text_input",
        },
    }

    _TYPE_TO_HANDLER = {}

    @classmethod
    def _initialize_type_mapping(cls):
        if cls._TYPE_TO_HANDLER:
            return

        for handler_name, type_set in cls._TYPE_HANDLERS.items():
            for type_name in type_set:
                cls._TYPE_TO_HANDLER[type_name] = handler_name

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_type_mapping()

    def _get_session_key(self, suffix=""):
        return f"{self.op.name}_{self.name}{suffix}"

    def _parse_list_value(self, value):
        if isinstance(value, list):
            return value

        if not value:
            return []

        value_str = str(value).strip()
        if not value_str:
            return []

        if value_str.startswith("[") and value_str.endswith("]"):
            parsed_value = ast.literal_eval(value_str)
            if isinstance(parsed_value, list):
                return parsed_value
            else:
                st.warning(f"Expected list format, actually is {type(parsed_value).__name__}")
                return []

        lines = [line.strip() for line in value_str.split("\n") if line.strip()]

        if "int" in self.type:
            try:
                converted_lines = []
                for line in lines:
                    converted_lines.append(int(line))
                lines = converted_lines
            except ValueError as e:
                st.warning(f"Integer conversion error: {str(e)}")
                return []
        elif "float" in self.type:
            try:
                converted_lines = []
                for line in lines:
                    converted_lines.append(float(line))
                lines = converted_lines
            except ValueError as e:
                st.warning(f"floating-point conversion error: {str(e)}")
                return []

        if self.type == "Union[str, List[str]]" and len(lines) == 1 and isinstance(lines[0], str):
            return lines[0]

        return lines

    def _on_v_change(self):
        new_v = st.session_state.get(self._get_session_key())

        handler_type = self._TYPE_TO_HANDLER.get(self.type)

        if handler_type in ("list_types"):
            new_v = self._parse_list_value(new_v)

        if handler_type == "dict_types":
            try:
                new_v = json.loads(new_v)
            except Exception:
                st.warning("The dictionary format entered is incorrect. Please check the JSON format.")
                new_v = {}

        if handler_type == "tuple_types":
            try:
                new_v = ast.literal_eval(new_v)
                if not isinstance(new_v, tuple):
                    st.warning(f"Expected tuple format, actually is {type(new_v).__name__}")
                    new_v = ()
            except Exception:
                st.warning("The tuple format entered is incorrect. Please check the tuple format.")
                new_v = ()

        try:
            self.set_v(new_v)
            self._update_quantile_if_needed()
        except Exception as e:
            logger.error(f"Error setting value: {e}")
            st.session_state[self._get_session_key()] = self.v

    def _on_p_change(self):
        new_p = st.session_state.get(self._get_session_key("_p"))
        try:
            self.set_p(new_p)
            st.session_state[self._get_session_key()] = self.v
        except Exception as e:
            logger.error(f"Error setting percentile: {e}")
            st.session_state[self._get_session_key("_p")] = self._v2p(self.v)

    def _update_quantile_if_needed(self):
        if self.stats_apply and self.quantiles is not None:
            st.session_state[self._get_session_key("_p")] = self._v2p(self.v)

    def _render_bool_input(self):
        return st.selectbox(
            self.name,
            options=self.v_options,
            key=self._get_session_key(),
            help=self.desc,
            on_change=self._on_v_change,
        )

    def _render_numeric_input(self):
        step = 1 if self.v_type == int else 0.01

        if self.stats_apply and self.quantiles is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.number_input(
                    self.name,
                    min_value=self.v_min,
                    max_value=self.v_max,
                    step=step,
                    key=self._get_session_key(),
                    help=self.desc,
                    on_change=self._on_v_change,
                )
            with col2:
                st.number_input(
                    "quantile",
                    min_value=0,
                    max_value=100,
                    step=1,
                    key=self._get_session_key("_p"),
                    on_change=self._on_p_change,
                )
        else:
            st.number_input(
                self.name,
                min_value=self.v_min,
                max_value=self.v_max,
                step=step,
                key=self._get_session_key(),
                help=self.desc,
                on_change=self._on_v_change,
            )

    def _render_string_input(self):
        if self.v_options is not None:
            return st.selectbox(
                self.name,
                options=self.v_options,
                key=self._get_session_key(),
                help=self.desc,
                on_change=self._on_v_change,
            )
        else:
            return st.text_input(
                label=self.name,
                key=self._get_session_key(),
                help=self.desc,
                on_change=self._on_v_change,
            )

    def _render_list_input(self, placeholder_text, help_suffix):
        if self.v_options is not None:
            return st.multiselect(
                self.name,
                options=self.v_options,
                key=self._get_session_key(),
                help=self.desc,
                on_change=self._on_v_change,
            )
        else:
            return st.text_area(
                self.name,
                placeholder=placeholder_text,
                key=self._get_session_key(),
                help=f"{self.desc}\n\n{help_suffix}",
                on_change=self._on_v_change,
            )

    def _render_dict_input(self, placeholder_text, help_suffix):

        return st.text_area(
            self.name,
            placeholder=placeholder_text,
            key=self._get_session_key(),
            help=f"{self.desc}\n\n{help_suffix}",
            on_change=self._on_v_change,
        )

    def _render_tuple_input(self):

        return st.text_input(
            self.name,
            placeholder="Enter Tuple (e.g. (5,))",
            key=self._get_session_key(),
            help=f"{self.desc}",
            on_change=self._on_v_change,
        )

    def _render_complex_type_input(self):
        config = self._COMPLEX_TYPE_CONFIGS.get(self.type)
        if not config:
            st.warning(f"Type '{self.type}' not fully supported, using text input")
            return st.text_input(
                f"{self.name} ({self.type})",
                key=self._get_session_key(),
                help=f"{self.desc}\n\nType: {self.type}",
                on_change=self._on_v_change,
            )

        component_func = st.text_area if config["component"] == "text_area" else st.text_input
        return component_func(
            self.name,
            placeholder=config["placeholder"],
            key=self._get_session_key(),
            help=f"{self.desc}\n\n{config['help']}",
            on_change=self._on_v_change,
        )

    def render(self):
        handler_type = self._TYPE_TO_HANDLER.get(self.type)

        if handler_type == "bool_types":
            self._render_bool_input()

        elif handler_type == "numeric_types":
            self._render_numeric_input()

        elif handler_type == "string_types":
            self._render_string_input()

        elif handler_type == "list_types":
            if "str" in self.type:
                placeholder = "Enter one item per line"
                help_text = "Enter one item per line"
            elif "int" in self.type:
                placeholder = "Enter one number per line\ne.g.:\n10\n20\n30"
                help_text = "Enter one number per line"
            elif "float" in self.type:
                placeholder = "Enter one number per line\ne.g.:\n1.5\n2.0\n3.14"
                help_text = "Enter one number per line"
            else:
                placeholder = "Enter one item per line"
                help_text = "Enter one item per line"

            self._render_list_input(placeholder, help_text)

        elif handler_type == "dict_types":
            self._render_dict_input(placeholder_text="Enter dictionary in JSON format", help_suffix="")
        elif handler_type == "tuple_types":
            self._render_tuple_input()

        else:
            self._render_complex_type_input()

    def st_sync(self):
        handler_type = self._TYPE_TO_HANDLER.get(self.type)

        if isinstance(self.v, list):
            if all(isinstance(item, str) for item in self.v):
                st_v = json.dumps(self.v, ensure_ascii=False)
            elif all(isinstance(item, (int, float)) for item in self.v):
                st_v = json.dumps(self.v)
            else:
                st_v = "\n".join(str(v) for v in self.v)
        elif handler_type == "numeric_types":
            st_v = self.v
        else:
            st_v = str(self.v)

        st.session_state[self._get_session_key()] = st_v
        if self.stats_apply and self.quantiles is not None:
            st.session_state[self._get_session_key("_p")] = self._v2p(self.v)


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
            st.checkbox("enabled", key=f"{self.name}_enabled", on_change=self.disable if self.enabled else self.enable)
            # render args
            if self.enabled:
                for arg_name in self.args:
                    self.args[arg_name].render()
                # Quantile plot in the end
                if self.quantiles is not None:
                    chart_data = pd.DataFrame(
                        np.array(self.quantiles).reshape(-1, 1),
                        columns=["quantile"],
                    )
                    st.line_chart(chart_data)

            if self.name == "language_id_score_filter":
                # display word cloud of language id
                if st.session_state.get("analyzed_dataset", None) is not None:
                    stats = st.session_state.analyzed_dataset[Fields.stats]
                    language_ids = [s[StatsKeys.lang] for s in stats]
                    fig, ax = plt.subplots()
                    text = " ".join(language_ids)
                    wordcloud = WordCloud(width=800, height=400, background_color="white", random_state=0).generate(
                        text
                    )
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)

    def st_sync(self):
        st.session_state[f"{self.name}_enabled"] = self.enabled
        for arg_name in self.args:
            self.args[arg_name].st_sync()


class StOperatorPool(OperatorPool):
    def __init__(self, config_path=None, default_ops=None):
        super(StOperatorPool, self).__init__(config_path=config_path, default_ops=default_ops)
        for op_name in self.pool:
            self.pool[op_name] = StOperator(self, state=self.pool[op_name].state)
        self.items_per_page = 8  # Number of operators displayed per page

        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
        if "search_term" not in st.session_state:
            st.session_state.search_term = ""

        recipes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs/data_juicer_recipes"))
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
        keys_to_delete = ["edited_op_pool_names", "edit_op_search_term"]
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]

    def add_ops_and_clear_callback(self, ops_to_add):
        st.session_state.edited_op_pool_names.update(ops_to_add)
        st.session_state.edit_op_search_term = ""

    @st.dialog("Edit Operator Pool", width="large")
    def render_edit_op_pool_dialog(self):
        # 1. Initialize temporary state on first open
        if "edited_op_pool_names" not in st.session_state:
            st.session_state.edited_op_pool_names = set(self.pool.keys())
        if "edit_op_current_page" not in st.session_state:
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
                if st.button(
                    "Remove All", key="remove_all_ops", disabled=not active_ops_list, use_container_width=True
                ):
                    st.session_state.edited_op_pool_names.clear()
                    st.rerun()

            if not active_ops_list:
                st.caption("No operators in the pool. Add some from the right!")

            with st.container(height=420):  # Adjusted height slightly
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
                            args=(op_name,),
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
                    args=(filtered_available_ops,),
                )

            ops_to_display = filtered_available_ops

            with st.container(height=350):  # Adjusted height
                if not ops_to_display:
                    st.caption("No matching operators found.")
                for op_name in ops_to_display:
                    row_col1, row_col2 = st.columns([0.8, 0.2])
                    with row_col1:
                        st.write(op_name)
                    with row_col2:
                        st.button(
                            "➕",
                            key=f"add_op_{op_name}",
                            help=f"Add {op_name}",
                            on_click=self.add_ops_and_clear_callback,
                            args=([op_name],),
                        )

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

                self.st_sync()  # IMPORTANT: Sync changes back to widget states
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
                    st.markdown("---")  # Visual separator

                    # Iterate through each operator and its configuration in the recipe
                    for op_name, op_config in recipe.operators.items():
                        st.markdown(f"**- {op_name}**")

                        # Check if there are arguments to display
                        args_from_recipe = op_config.get("args", {})
                        if args_from_recipe:
                            for arg_name, arg_details in args_from_recipe.items():
                                param_col1, param_col2 = st.columns([0.4, 0.6])

                                with param_col1:
                                    st.caption(f"`{arg_name}`")

                                with param_col2:
                                    v = arg_details.get("v")
                                    if v is not None:
                                        display_v = v
                                    else:
                                        display_v = arg_details.get("default")

                                    if isinstance(display_v, list):
                                        display_v = f"[{', '.join(map(str, display_v))}]"
                                    st.caption(f": {display_v}")
                        else:
                            st.caption("  *(no parameters)*")

                        st.text("")  # Add a little vertical space between operators

                    st.markdown("---")
                    if st.button(
                        "Apply this Recipe", key=f"apply_{recipe.path}", type="primary", use_container_width=True
                    ):
                        logger.info(f"Applying recipe: {recipe.name}")

                        # 1. Clear the current operator pool
                        self.pool.clear()

                        # 2. Add all operators from the selected recipe
                        for op_name, op_config_to_add in recipe.operators.items():
                            self.add_op(op_name, op_config_to_add)
                            for arg_name, arg_details in op_config_to_add.get("args", {}).items():
                                if arg_details.get("v") is not None:
                                    self.act(
                                        op_name=op_name, action_type="set_arg", arg_name=arg_name, v=arg_details["v"]
                                    )

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
                    export_path=st.session_state.get("export_path", "./data/processed_dataset.jsonl"),
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
                if st.button("→", disabled=self.current_page >= total_pages):
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
