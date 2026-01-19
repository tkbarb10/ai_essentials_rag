from typing import Dict, Any, List, Union, Optional
from utils.logging_helper import setup_logging
from utils.load_yaml_config import load_yaml_config
from config.paths import COMPONENTS_FPATH

# Initialize logger for this module
logger = setup_logging(name='prompt_builder')

# Default component values - used when components are missing or None
comp_options = load_yaml_config(COMPONENTS_FPATH)


def format_prompt_section(lead_in: str, value: Union[str, List[str]]) -> str:
    """Format a prompt section by joining a lead-in with content.

    If `value` is a list, each item will be rendered as a bullet line ("- item").

    Args:
        lead_in: Lead-in sentence for the section.
        value: Section content, either a single string or list of strings.

    Returns:
        A single string consisting of the `lead_in` followed by the formatted content.
    """
    if isinstance(value, list):
        formatted_value = "\n".join(f"- {item}" for item in value)
    else:
        formatted_value = value
    return f"{lead_in}\n{formatted_value}"


def build_prompt(
    config: Dict[str, Any],
    context: Optional[List[str]] = None,
    strategy: Optional[str] = None,
    topic: Optional[str] = None,
    categories: Optional[List[str]] = None,
    components: Optional[Dict[str, Any]] = None
) -> str:
    """Assemble a system prompt from configuration and optional context.

    The function expects `config` to include an 'instructions' key (ValueError is
    raised if missing). Optional fields such as 'categories', 'constraints',
    'tools', 'tone', 'format', 'goal', and 'reasoning_strategies' are supported.

    NEW: Supports {topic} and {categories} placeholders in prompts for dynamic
    substitution. Components (tone, reasoning_strategies, tools) can be provided
    separately to enable reusable prompt templates.

    Args:
        config: Prompt configuration mapping containing required 'instructions'
            and optional metadata fields.
        context: Optional list of context strings appended to the prompt (rendered
            as bulleted lines).
        strategy: Optional key selecting a reasoning strategy from the config or
            components. Defaults to 'Self-Ask' if not provided.
        topic: Optional topic name to replace {topic} placeholders in the prompt.
            If not provided, {topic} placeholders remain unchanged.
        categories: Optional list of categories to format and replace {categories}
            placeholder in instructions. Used primarily for prep_prompt.
        components: Optional dict containing reusable components:
            - 'tones': Dict of named tone configurations
            - 'reasoning_strategies': Dict of named reasoning strategies
            - 'tools': List of tool descriptions
            If None, defaults are applied: conversational tone, Self-Ask reasoning,
            tools enabled.

    Returns:
        Fully assembled prompt string with all placeholders replaced.

    Raises:
        ValueError: If the 'instructions' key is missing from the config.

    Example:
        >>> config = {'instructions': 'Answer questions about {topic}'}
        >>> components = {'tones': {'conversational': ['Be friendly']}}
        >>> prompt = build_prompt(config, topic='Healthcare', components=components)
    """
    prompt_parts = []

    # Get role, with {topic} placeholder support
    role = config.get("role", "Helpful assistant that answers questions about {topic}")

    # Replace {topic} if provided
    if topic:
        role = role.replace("{topic}", topic)

    prompt_parts.append(f"You are a {role.strip()}")

    # Instructions are required
    instructions = config.get('instructions')
    if not instructions:
        raise ValueError("Missing the 'instructions' field, this bot needs direction")

    # Replace {topic} placeholder in instructions
    if topic:
        instructions = instructions.replace("{topic}", topic)

    # Replace {categories} placeholder in instructions
    if categories:
        # Format categories as a bulleted list
        categories_formatted = "\n".join(f"- {cat}" for cat in categories)
        instructions = instructions.replace("{categories}", categories_formatted)

    prompt_parts.append(instructions.strip())

    # Handle categories from config (legacy support for hardcoded categories)
    if categories_from_config := config.get("categories"):
        prompt_parts.append(
            format_prompt_section(
                "Organizational Categories:", categories_from_config
            )
        )

    # Constraints
    if constraints := config.get('constraints'):
        # Replace {topic} in constraints
        if topic:
            constraints = [c.replace("{topic}", topic) for c in constraints]
        prompt_parts.append(
            format_prompt_section(
                "Output constraints:", constraints
            )
        )

    # Tools
    if components:
        try:
            # Tools component is a list directly, not nested under 'description'
            if components.get('tools'):
                tools_content = comp_options['tools']
                prompt_parts.append(
                    format_prompt_section(
                        "Tools:", tools_content
                    )
                )
                logger.info(f"Using these tools from components: {tools_content}")
        except Exception as e:
            print(f"Warning: Error loading tools from components: {e}")
            logger.warning(f"Error loading tools from components: {e}, falling back to config")

    # Tone
    tone_content = None
    if components:
        try:
            if 'tones' in components:
                tone_key = components.get('tones')
                tone_content = comp_options.get('tones').get(tone_key) # type: ignore
                logger.info(f"Loaded tone '{tone_key}' from components")
        except Exception as e:
            print(f"Warning: Error loading tone from components: {e}")
            logger.warning(f"Error loading tone from components: {e}, falling back to config")
    
    if not tone_content:
        tone_content = [
            'Personal and human',
            'Avoid buzzwords',
            "Don't use em dashes or semicolons",
            'Favor brevity over long, compound sentences'
            ]

    prompt_parts.append(
        format_prompt_section(
            "Communication Style:", tone_content
        )
    )

    # Format
    if format_section := config.get("format"):
        prompt_parts.append(
            format_prompt_section(
                "Output Format:", format_section
            )
        )

    # Goal with {topic} replacement
    if goal := config.get("goal"):
        if topic:
            goal = goal.replace("{topic}", topic)
        prompt_parts.append(f"The goal of this interaction is: {goal.strip()}")

    # Reasoning strategy - try components first, then config, with default fallback
    if strategy is None:
        strategy = 'Self-Ask'

    reasoning_content = None
    if strategy:
        if components:
            try:
                # Look for reasoning strategies in components
                if 'reasoning_strategies' in components:
                    reasoning_key = components['reasoning_strategies']
                    reasoning_content = comp_options.get("reasoning_strategies").get(reasoning_key) # type: ignore
                    logger.info(f"Loaded reasoning strategy '{strategy}' from components")
            except Exception as e:
                print(f"Warning: Error loading reasoning strategy from components: {e}")
                logger.warning(f"Error loading reasoning strategy from components: {e}, falling back to config")

        # Fall back to config if not found in components
        if reasoning_content is None:
            if reasoning_strategies := comp_options.get("reasoning_strategies"):
                reasoning_content = reasoning_strategies.get(strategy)

        if reasoning_content:
            prompt_parts.append(reasoning_content)

    # Context (for RAG)
    if context is not None:
        context_string = "\n".join(f"-{item}" for item in context)
        prompt_parts.append(
            "Here is additional context to help with your answers\n\n"
            "=== ADDITIONAL CONTEXT ===\n"
            f"{context_string}\n"
            "=== END ADDITIONAL CONTEXT ===\n\n"
            "=== PREVIOUS CONVERSATION HISTORY ===\n"
        )

        logger.info(f"=== How Prompt with context added looks ===\n\n{"\n\n".join(prompt_parts)}")

    return "\n\n".join(prompt_parts)