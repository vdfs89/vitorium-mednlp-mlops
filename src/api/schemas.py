"""Pydantic schemas for the VITORIUM inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LaudoInput(BaseModel):
    """Input payload: a single medical report/abstract to triage."""

    texto: str = Field(
        ...,
        min_length=10,
        max_length=20_000,
        description=(
            "Texto do laudo/relato clinico a ser classificado (espacos nas pontas "
            "sao removidos antes de validar o tamanho; precisa conter letras)."
        ),
    )

    @field_validator("texto", mode="before")
    @classmethod
    def strip_whitespace(cls, value: object) -> object:
        """Trim before the length check, so 20 spaces do not count as 20 characters."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("texto")
    @classmethod
    def require_letters(cls, value: str) -> str:
        """Reject texts with no letters (only punctuation/digits/symbols).

        Such inputs have an all-zero TF-IDF vector, so the model can only return
        its intercept-driven default (``normal`` at ~0.41) -- an answer that
        looks like a prediction but carries no information.
        """
        if not any(ch.isalpha() for ch in value):
            raise ValueError("texto precisa conter palavras (apenas simbolos/numeros)")
        return value

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "texto": (
                        "Patient presents with acute chest pain radiating to the left "
                        "arm, elevated troponin levels, and ECG changes consistent "
                        "with myocardial infarction."
                    )
                }
            ]
        }
    }


class PredictionOutput(BaseModel):
    """Output payload: predicted urgency tier and model confidence."""

    urgencia: str = Field(..., description="normal | atencao | urgente")
    confianca: float = Field(..., ge=0.0, le=1.0)
    probabilidades: dict[str, float] = Field(
        ..., description="Probabilidade por classe."
    )
    latencia_ms: float = Field(..., description="Tempo de inferencia em milissegundos.")


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool
    backend: str = Field(
        ..., description="Motor de inferencia ativo: onnx | sklearn | none (sem modelo)."
    )
