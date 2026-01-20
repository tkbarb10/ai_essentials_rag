"""Pytest configuration and shared fixtures for the test suite."""
import sys
from pathlib import Path
import shutil
import tempfile
import uuid
import pytest

# Ensure project root is on sys.path so imports like `utils` and `rag_assistant` work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# If optional external dependencies are not installed in the environment, insert
# lightweight stubs so test collection succeeds. Tests that require the real
# libraries will skip at runtime if the real package or API key is not available.

# Stub for torch (required by vector_store.initialize)
try:
    import torch  # type: ignore
except ModuleNotFoundError:
    import types
    _torch = types.ModuleType('torch')
    _torch.__stub__ = True  # type: ignore
    _torch.cuda = types.ModuleType('torch.cuda')  # type: ignore
    _torch.cuda.is_available = lambda: False  # type: ignore
    _torch.backends = types.ModuleType('torch.backends')  # type: ignore
    _torch.backends.mps = types.ModuleType('torch.backends.mps')  # type: ignore
    _torch.backends.mps.is_available = lambda: False  # type: ignore
    sys.modules['torch'] = _torch
    sys.modules['torch.cuda'] = _torch.cuda
    sys.modules['torch.backends'] = _torch.backends
    sys.modules['torch.backends.mps'] = _torch.backends.mps

# Stub for langchain_text_splitters
try:
    import langchain_text_splitters  # type: ignore
except ModuleNotFoundError:
    import types
    _lts = types.ModuleType('langchain_text_splitters')
    _lts.__stub__ = True  # type: ignore

    class _Document:
        def __init__(self, page_content='', metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class _MarkdownHeaderTextSplitter:
        __stub__ = True
        def __init__(self, headers_to_split_on=None, **kw):
            self.headers = headers_to_split_on or []
        def split_text(self, text):
            # Return a simple document with the full text
            return [_Document(page_content=text, metadata={})]

    class _RecursiveCharacterTextSplitter:
        __stub__ = True
        def __init__(self, chunk_size=1000, chunk_overlap=100, **kw):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        def split_documents(self, docs):
            # Just return the docs as-is for stub purposes
            return docs

    _lts.MarkdownHeaderTextSplitter = _MarkdownHeaderTextSplitter  # type: ignore
    _lts.RecursiveCharacterTextSplitter = _RecursiveCharacterTextSplitter  # type: ignore
    sys.modules['langchain_text_splitters'] = _lts

try:
    import tavily  # type: ignore
except ModuleNotFoundError:
    import types
    tavily = types.ModuleType('tavily')
    class _TavilyStub:
        __stub__ = True
        def __init__(self, *a, **kw):
            pass
        def map(self, *a, **kw):
            raise RuntimeError("Missing package 'tavily'. Install it to run Tavily integration tests")
        def extract(self, *a, **kw):
            raise RuntimeError("Missing package 'tavily'. Install it to run Tavily integration tests")
    tavily.TavilyClient = _TavilyStub # type: ignore
    sys.modules['tavily'] = tavily

try:
    import langchain_huggingface  # type: ignore
except ModuleNotFoundError:
    import types
    _hf = types.ModuleType('langchain_huggingface')
    class _HFStub:
        __stub__ = True
        def __init__(self, *a, **kw):
            pass
    _hf.HuggingFaceEmbeddings = _HFStub # type: ignore
    sys.modules['langchain_huggingface'] = _hf

try:
    import langchain_chroma  # type: ignore
except ModuleNotFoundError:
    import types
    _ch = types.ModuleType('langchain_chroma')
    class _ChromaStub:
        __stub__ = True
        def __init__(self, persist_path=None, collection_name=None, embedding_model=None, **kw):
            self._docs = []
            self.persist_path = persist_path
            self.collection_name = collection_name
        def add_documents(self, documents):
            # Accept LangChain Documents or dict-likes
            self._docs.extend(documents)
        def search(self, query, search_type='similarity', k=3):
            # naive substring match on page_content
            results = []
            q = str(query).lower()
            for doc in self._docs:
                content = getattr(doc, 'page_content', str(doc))
                if q in content.lower():
                    results.append(doc)
            # return up to k results
            return results[:k]
    _ch.Chroma = _ChromaStub # type: ignore
    sys.modules['langchain_chroma'] = _ch

try:
    import langchain.chat_models  # type: ignore
except ModuleNotFoundError:
    import types
    _cm = types.ModuleType('langchain.chat_models')
    _cm.__stub__ = True  # type: ignore
    def init_chat_model(**kwargs):
        raise RuntimeError("Missing 'langchain' chat model provider. Install the required provider to run LLM integration tests")
    _cm.init_chat_model = init_chat_model # type: ignore
    sys.modules['langchain.chat_models'] = _cm

try:
    import groq  # type: ignore
except ModuleNotFoundError:
    import types
    _groq = types.ModuleType('groq')
    class _GroqStub:
        __stub__ = True
        def __init__(self, *a, **kw):
            pass
    _groq.Groq = _GroqStub # type: ignore
    sys.modules['groq'] = _groq

# Integration tests use real APIs and keys loaded from .env
from config.load_env import load_env


@pytest.fixture(scope="session")
def env_config():
    """Load and return environment configuration for tests.

    Tests can selectively skip themselves if the required keys are not present.
    Also annotate whether optional provider libraries are present (not stubs).
    """
    env = load_env()

    # Detect whether optional libraries are present or only provided by the test stubs
    def _has_real_mod(name):
        mod = sys.modules.get(name)
        return bool(mod) and not getattr(mod, "__stub__", False)

    env["HAS_TAVILY"] = _has_real_mod("tavily")
    env["HAS_LANGCHAIN_CHROMA"] = _has_real_mod("langchain_chroma")
    env["HAS_LANGCHAIN"] = _has_real_mod("langchain.chat_models")
    env["HAS_GROQ"] = _has_real_mod("groq")

    return env


@pytest.fixture
def test_chroma_dir(tmp_path):
    """Provide a unique chroma persist directory for each test and ensure cleanup.

    Yields a string path usable as `persist_path` by the application code.
    """
    import time
    import gc

    unique = f"test_chroma_{uuid.uuid4().hex[:8]}"
    path = Path("./chroma") / unique
    path.mkdir(parents=True, exist_ok=True)
    yield str(path)

    # Teardown: remove the directory if it still exists
    # On Windows, Chroma may hold file locks; retry with delay
    if path.exists():
        gc.collect()  # Help release any lingering references
        for attempt in range(3):
            try:
                shutil.rmtree(path)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.5)
                # On final attempt, ignore the error - cleanup will happen eventually

