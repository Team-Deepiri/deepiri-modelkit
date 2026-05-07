from deepiri_modelkit.streaming.topics import StreamTopics


def test_document_stream_topics_match_lis_routing_namespace() -> None:
    assert StreamTopics.DOCUMENT_VECTORIZE.value == "document.vectorize"
    assert StreamTopics.DOCUMENT_TRAINING.value == "document.training"
    assert StreamTopics.DOCUMENT_STRUCTURED.value == "document.structured"
    assert StreamTopics.DOCUMENT_ARTIFACTS.value == "document.artifacts"


def test_helox_training_topics_stay_in_pipeline_namespace() -> None:
    assert StreamTopics.HELOX_TRAINING_RAW.value == "pipeline.helox-training.raw"
    assert (
        StreamTopics.HELOX_TRAINING_STRUCTURED.value
        == "pipeline.helox-training.structured"
    )


def test_all_includes_shared_stream_topics() -> None:
    topics = set(StreamTopics.all())

    assert "document.vectorize" in topics
    assert "document.training" in topics
    assert "document.structured" in topics
    assert "document.artifacts" in topics
    assert "pipeline.helox-training.raw" in topics
    assert "pipeline.helox-training.structured" in topics
