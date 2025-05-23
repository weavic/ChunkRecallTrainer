# conftest.py
import pytest
from langchain_openai import ChatOpenAI


@pytest.fixture(autouse=True)
def patch_chat_openai(monkeypatch):
    class DummyLLM(ChatOpenAI):
        def invoke(self, *args, **kwargs):
            if "practice question" in str(args[0]):
                return {"question": "Use 'How are you?'", "answer": "How are you?"}
            return {"score": 4, "better": None, "comment": "Perfect!"}

    monkeypatch.setattr("chunk_recall_trainer.graph.ChatOpenAI", DummyLLM)
