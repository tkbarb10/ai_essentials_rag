"""Type definitions for configuration and component settings.

This module provides TypedDict definitions for type-hinting configuration
dictionaries used throughout the application, particularly for RAG assistant
prompt components.
"""

from typing import Literal, TypedDict


ToneType = Literal['conversational', 'professional', 'technical']
ReasoningStrategyType = Literal['CoT', 'ReAct', 'Self-Ask']


class ComponentsDict(TypedDict, total=False):
    """Type definition for RAG assistant prompt components.

    Attributes:
        tones: Communication style for responses. One of 'conversational',
            'professional', or 'technical'.
        reasoning_strategies: Reasoning approach for the LLM. One of 'CoT'
            (Chain of Thought), 'ReAct', or 'Self-Ask'.
        tools: Whether to enable tool descriptions in the prompt.
    """
    tones: ToneType
    reasoning_strategies: ReasoningStrategyType
    tools: bool
