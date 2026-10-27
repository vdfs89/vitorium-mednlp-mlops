"""Pydantic schemas for the VITORIUM inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LaudoInput(BaseModel):
    """Input payload: a single medical report/abstract to triage."""

    texto: str = Field(
        ...,
        min_length=10,
        max_length=20_000,
        description="Texto do laudo/relato clinico a ser classificado.",
    )

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
