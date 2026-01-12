import yaml
from pathlib import Path
from typing import Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary (mapping). Note: `yaml.safe_load`
        may return `None` for empty files.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML is invalid.
        IOError: If the file cannot be read.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"YAML config not found as {file_path}")

    try:
        with open(file_path, "r", encoding='utf-8') as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file, check the syntax in the file: {e}") from e
    except IOError as e:
        raise IOError(f"Error reading YAML file, try again: {e}") from e


def load_all_prompts(prompts_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load all YAML files from the prompts directory and merge them.

    This function loads rag_prompts.yaml, ingestion_prompts.yaml, and components.yaml
    from the specified directory and merges them into a single configuration dictionary.

    Args:
        prompts_dir: Path to the directory containing prompt YAML files.

    Returns:
        Combined configuration dictionary with keys from all three files:
        - Prompt templates from rag_prompts.yaml and ingestion_prompts.yaml
        - 'tones', 'reasoning_strategies', 'tools' from components.yaml

    Raises:
        FileNotFoundError: If the prompts directory or required files don't exist.
        yaml.YAMLError: If any YAML file is invalid.

    Example:
        >>> config = load_all_prompts('./prompts')
        >>> qa_prompt = config['qa_assistant']
        >>> tone = config['tones']['conversational']
    """
    prompts_dir = Path(prompts_dir)

    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    combined_config = {}

    # Load each YAML file and merge into combined config
    yaml_files = {
        'rag_prompts.yaml': 'RAG prompt templates',
        'ingestion_prompts.yaml': 'Ingestion prompt templates',
        'components.yaml': 'Reusable components'
    }

    for filename, description in yaml_files.items():
        file_path = prompts_dir / filename

        try:
            if file_path.exists():
                logger.info(f"Loading {description} from {file_path}")
                config = load_yaml_config(file_path)
                if config:
                    combined_config.update(config)
            else:
                logger.warning(f"Optional file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            # Continue loading other files even if one fails

    return combined_config