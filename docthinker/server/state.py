from dataclasses import dataclass
from typing import Optional, Any

from docthinker.session_manager import SessionManager
from docthinker.cognitive import CognitiveProcessor
from docthinker.services import IngestionService
from docthinker.providers import AppSettings


@dataclass
class AppState:
    settings: Optional[AppSettings] = None
    api_config: Optional[Any] = None
    session_manager: Optional[SessionManager] = None

    rag_instance: Optional[Any] = None
    cognitive_processor: Optional[CognitiveProcessor] = None
    ingestion_service: Optional[IngestionService] = None
    orchestrator: Optional[Any] = None
    memory_engine: Optional[Any] = None  # neuro_memory.MemoryEngine，类人脑联想与记忆重连


state = AppState()
