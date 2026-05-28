from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Source:
    document: str
    page: int | str

    @property
    def citation(self) -> str:
        return f"{self.document} (Page {self.page})"


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[str]
    contexts: list[str]
    metadata: list[dict[str, Any]]

