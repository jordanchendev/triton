import pytest
from pydantic import ValidationError

from triton.schemas import TaskCreate, DocumentCreate, ScheduleCreate


def test_task_create_with_url():
    task = TaskCreate(type="youtube", source_url="https://youtube.com/watch?v=xxx")
    assert task.type == "youtube"
    assert task.source_url == "https://youtube.com/watch?v=xxx"


def test_task_create_invalid_type():
    with pytest.raises(ValidationError):
        TaskCreate(type="invalid_type", source_url="https://example.com")


def test_document_create():
    doc = DocumentCreate(type="web", content="Hello world", title="Test")
    assert doc.content == "Hello world"


def test_schedule_create():
    sched = ScheduleCreate(
        name="Daily YouTube",
        cron_expression="0 8 * * *",
        type="youtube",
        config={"channel_ids": ["UC123"]},
    )
    assert sched.cron_expression == "0 8 * * *"
