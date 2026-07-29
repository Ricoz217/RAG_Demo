import pytest

from rag_demo.application import ApplicationNotStartedError, RAGApplication
from rag_demo.config import Settings


def test_application_rejects_resource_access_before_start() -> None:
    application = RAGApplication(Settings())

    with pytest.raises(ApplicationNotStartedError, match="not started"):
        _ = application.database
