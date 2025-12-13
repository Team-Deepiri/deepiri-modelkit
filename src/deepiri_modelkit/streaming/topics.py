"""
Stream topic definitions
"""
from enum import Enum


class StreamTopics(str, Enum):
    """Redis Stream topics"""
    MODEL_EVENTS = "model-events"
    INFERENCE_EVENTS = "inference-events"
    PLATFORM_EVENTS = "platform-events"
    AGI_DECISIONS = "agi-decisions"
    TRAINING_EVENTS = "training-events"
    
    @classmethod
    def all(cls) -> list:
        """Get all topic names"""
        return [topic.value for topic in cls]

