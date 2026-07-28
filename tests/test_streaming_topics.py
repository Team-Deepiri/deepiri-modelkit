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


def test_agi_pipeline_bus_topics() -> None:
    assert StreamTopics.PIPELINE_PRESSURE_EVENTS.value == "pipeline.pressure.events"
    assert (
        StreamTopics.PIPELINE_ARTIFACT_INVALIDATION.value
        == "pipeline.artifact.invalidation"
    )
    assert StreamTopics.PIPELINE_SPLICE_EVENTS.value == "pipeline.splice.events"
    assert StreamTopics.PIPELINE_DEAD_LETTER.value == "pipeline.dead-letter"
    assert StreamTopics.PIPELINE_METRICS.value == "pipeline.metrics"
    assert StreamTopics.TRAINING_JOBS.value == "training-jobs"


def test_all_includes_shared_stream_topics() -> None:
    topics = set(StreamTopics.all())

    assert "document.vectorize" in topics
    assert "document.training" in topics
    assert "document.structured" in topics
    assert "document.artifacts" in topics
    assert "pipeline.helox-training.raw" in topics
    assert "pipeline.helox-training.structured" in topics
    assert "pipeline.pressure.events" in topics
    assert "pipeline.artifact.invalidation" in topics
    assert "pipeline.splice.events" in topics
    assert "training-jobs" in topics
    assert len(StreamTopics.sugar_glider_allowlist()) == len(topics)
