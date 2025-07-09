import os
import copy
import yaml
from loguru import logger

class Recipe:
    """A simple data class to hold processed recipe information."""
    def __init__(self, name: str, path: str, operators: dict, description: str = ""):
        self.name = name
        self.path = path
        # operators is a dictionary: {op_name: op_config_dict}
        self.operators = operators
        self.description = description


class RecipeManager:
    """Finds, parses, and converts recipe files into a usable format."""

    def __init__(self, recipes_dir: str, all_ops_config: dict):
        self.recipes_dir = recipes_dir
        self.all_ops = all_ops_config
        self.recipes = self._load_all_recipes()

    def _load_all_recipes(self) -> list[Recipe]:
        """Recursively finds and processes all recipe .yaml files."""
        loaded_recipes = []
        if not os.path.isdir(self.recipes_dir):
            logger.warning(f"Recipes directory not found: {self.recipes_dir}")
            return []
            
        for root, _, files in os.walk(self.recipes_dir):
            if "sandbox" in root:
                continue
            for filename in files:
                if filename.endswith(".yaml"):
                    filepath = os.path.join(root, filename)
                    recipe = self._parse_and_convert(filepath, filename)
                    if recipe:
                        loaded_recipes.append(recipe)
        
        # Sort recipes by name for consistent display
        loaded_recipes.sort(key=lambda r: r.name)
        return loaded_recipes

    def _parse_and_convert(self, filepath: str, filename: str) -> Recipe | None:
        """Parses a single YAML file and converts it to a Recipe object."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_recipe = yaml.safe_load(f)

            if not raw_recipe or 'process' not in raw_recipe or not isinstance(raw_recipe['process'], list):
                # Skip files that aren't valid recipes
                return None

            converted_ops = {}
            for op_entry in raw_recipe['process']:
                if not isinstance(op_entry, dict) or len(op_entry) != 1:
                    logger.warning(f"Skipping malformed entry in {filename}: {op_entry}")
                    continue
                
                op_name = list(op_entry.keys())[0]
                op_args_from_recipe = op_entry[op_name] or {}
                logger.info(f"Converting operator '{op_name}', args: {op_args_from_recipe}")

                if op_name not in self.all_ops:
                    logger.warning(f"Recipe '{filename}' uses operator '{op_name}' which is not in all_ops.yaml. Skipping this operator.")
                    continue

                base_op_config = copy.deepcopy(self.all_ops[op_name])

                # Update: update parameter values in recipe to 'v' field
                if base_op_config.get('args') and op_args_from_recipe:
                    for arg_name, arg_value in op_args_from_recipe.items():
                        if arg_name in base_op_config['args']:
                            base_op_config['args'][arg_name]['v'] = arg_value
                        else:
                            logger.warning(f"In recipe '{filename}', operator '{op_name}' has an unknown argument '{arg_name}'. Ignoring.")
                
                converted_ops[op_name] = base_op_config
            
            if not converted_ops:
                # If after processing, no valid operators were found, don't create a recipe
                return None

            # Generate a user-friendly name from the filename
            recipe_name = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()
            
            return Recipe(name=recipe_name, path=filepath, operators=converted_ops)

        except Exception as e:
            logger.error(f"Failed to load or parse recipe {filepath}: {e}")
            return None

