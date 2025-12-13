"""
Streaming event schemas and validation
"""
from .topics import StreamTopics
from ..contracts.events import (
    BaseEvent,
    ModelReadyEvent,
    ModelLoadedEvent,
    InferenceEvent,
    PlatformEvent,
    AGIDecisionEvent,
    TrainingEvent,
)


# Map topics to event schemas
TOPIC_EVENT_SCHEMAS = {
    StreamTopics.MODEL_EVENTS: [ModelReadyEvent, ModelLoadedEvent],
    StreamTopics.INFERENCE_EVENTS: [InferenceEvent],
    StreamTopics.PLATFORM_EVENTS: [PlatformEvent],
    StreamTopics.AGI_DECISIONS: [AGIDecisionEvent],
    StreamTopics.TRAINING_EVENTS: [TrainingEvent],
}


def validate_event(topic: str, event_data: dict) -> BaseEvent:
    """
    Validate event against schema
    
    Args:
        topic: Stream topic
        event_data: Event data dict
    
    Returns:
        Validated event object
    
    Raises:
        ValueError: If event doesn't match schema
    """
    if topic not in TOPIC_EVENT_SCHEMAS:
        # Unknown topic, return base event
        return BaseEvent(**event_data)
    
    schemas = TOPIC_EVENT_SCHEMAS[topic]
    event_type = event_data.get("event")
    
    # Try to match event type to schema
    for schema in schemas:
        try:
            return schema(**event_data)
        except Exception:
            continue
    
    # Fallback to base event
    return BaseEvent(**event_data)

