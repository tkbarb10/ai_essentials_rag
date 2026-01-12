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
    tavily.TavilyClient = _TavilyStub
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
    _hf.HuggingFaceEmbeddings = _HFStub
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
    _ch.Chroma = _ChromaStub
    sys.modules['langchain_chroma'] = _ch

try:
    import langchain.chat_models  # type: ignore
except ModuleNotFoundError:
    import types
    _cm = types.ModuleType('langchain.chat_models')
    def init_chat_model(**kwargs):
        raise RuntimeError("Missing 'langchain' chat model provider. Install the required provider to run LLM integration tests")
    _cm.init_chat_model = init_chat_model
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
    _groq.Groq = _GroqStub
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
    env["HAS_GROQ"] = _has_real_mod("groq")

    return env


@pytest.fixture
def test_chroma_dir(tmp_path):
    """Provide a unique chroma persist directory for each test and ensure cleanup.

    Yields a string path usable as `persist_path` by the application code.
    """
    unique = f"test_chroma_{uuid.uuid4().hex[:8]}"
    path = Path("./chroma") / unique
    path.mkdir(parents=True, exist_ok=True)
    yield str(path)

    # Teardown: remove the directory if it still exists
    if path.exists():
        shutil.rmtree(path)

