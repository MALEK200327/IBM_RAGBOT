"""
Test module for unit tests on RAG.py
"""
import pytest
from unittest.mock import Mock, patch
import sys
sys.path.append('C:/Users/najla/Documents/CPC/College/Sheffield/Year 2/Semester 2/COM21002 AI Group Project/RagBot/src/')
from RAG import RAG

rag = RAG("C:/Users/najla/Documents/CPC/College/Sheffield/Year 2/Semester 2/COM21002 AI Group Project/RagBot/resources/Test.pdf")

@pytest.fixture
def setup_rag():
    return RAG("C:/Users/najla/Documents/CPC/College/Sheffield/Year 2/Semester 2/COM21002 AI Group Project/RagBot/resources/Test.pdf")


def test_loadpdf():
    # Test if the document is loaded
    assert rag.pages is not None
    assert rag.pages[0] is not None

def test_split():
    # Test if the document is propperly split into pages
    assert len(rag.pages) > 1
    print(rag.pages[0])
    print(rag.pages[1])

def test_prompts():
    # Test if the prompts are initialized
    assert rag.recommend_course_prompt is not None
    assert rag.infer_context_prompt is not None
    assert rag.question_and_answer_prompt is not None
    assert rag.further_questions_prompt is not None

def test_retriever():
    # Test if the retriever is initialized
    assert rag.retriever is not None

def test_model():
    # Test if the model is initialized
    assert rag.model is not None

def test_infer_context(setup_rag):
    """Test infer_context method."""
    with patch.object(setup_rag, 'augment', return_value="augmented prompt") as mock_augment, \
         patch.object(setup_rag, 'generate', return_value="3") as mock_generate:
        answers = {"Question1": "Engineering", "Question2": "Engineering", "Question3": "Engineering", "Question4": "Engineering", "Question5": "Engineering"}
        context = setup_rag.infer_context(answers)
        assert context == "IBM AUTOMATION"
        mock_augment.assert_called()
        mock_generate.assert_called_once()

def test_retrieve(setup_rag):
    """Test retrieve method."""
    with patch('langchain_community.vectorstores.Chroma.from_documents') as mock_chroma:
        setup_rag.__init_retriever__()
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.return_value = [Mock(page_content="relevant content\nzblithereplz\nmore content")]
        setup_rag.retriever = mock_retriever
        document = setup_rag.retrieve("test query")
        assert "relevant content more content" in document

def test_augment(setup_rag):
    """Test augment method."""
    template = "Hello, {}!"
    args = ("world",)
    result = setup_rag.augment(template, *args)
    assert result == "Hello, world!"

