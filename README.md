<p align="center">
  <img src="docs/cover.svg" alt="VITORIUM — End-to-End MLOps Pipeline for Medical Text Classification" width="100%">
</p>

# VITORIUM

**End-to-End MLOps Pipeline for Medical Text Classification**
Medical NLP · MLOps · Production ML

Tech Challenge — FIAP Pós-Tech em Machine Learning Engineering, Fase 3 (Cloud and MLOps).

> Especificação técnica completa, com rastreabilidade de requisitos e ADRs,
> em [`docs/SPEC.md`](docs/SPEC.md) — mantenha-o atualizado a cada etapa.
> Regra de versionamento (tags Git) em [`docs/VERSIONING.md`](docs/VERSIONING.md).

## Contexto

Um hospital de referência precisa de um sistema de triagem automática de laudos
médicos para classificar urgência. O VITORIUM é um classificador de texto (NLP)
leve, servido via API REST em container Docker, com pipeline de retreino
orquestrado (Airflow), monitoramento (Prometheus + Grafana) e otimização de
latência (ONNX Runtime).

## Dataset e mapeamento de rótulos — nota de honestidade

Usamos o [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
(sebischair, NLPIR 2022): 14.438 abstracts médicos rotulados em 5 categorias de
condição de paciente.

| Classe original | Treino | Teste | Total |
|---|---:|---:|---:|
| 1. Neoplasms | 2.530 | 633 | 3.163 |
| 2. Digestive system diseases | 1.195 | 299 | 1.494 |
| 3. Nervous system diseases | 1.540 | 385 | 1.925 |
| 4. Cardiovascular diseases | 2.441 | 610 | 3.051 |
| 5. General pathological conditions | 3.844 | 961 | 4.805 |

O corpus não traz rótulos de urgência clínica. Aplicamos um **mapeamento
heurístico** (`src/training/preprocess.py`) para a narrativa de triagem do
desafio:

| Urgência | Classes originais | Amostras | % |
|---|---|---:|---:|
| `urgente` | Cardiovascular + Nervous system | 4.976 | 34,5% |
| `atencao` | Neoplasms + Digestive system | 4.657 | 32,3% |
| `normal` | General pathological conditions | 4.805 | 33,3% |

> **Os rótulos de urgência (normal/atenção/urgente) são um mapeamento
> heurístico das 5 categorias originais do dataset, criado para fins
> didáticos de MLOps. Não refletem triagem clínica real (ex.: Manchester,
> ESI) e não devem ser usados para decisões médicas.**

**Atribuição:** dataset licenciado sob CC BY-SA 3.0. Autoria original:
Schopf, T., Braun, D., Matthes, F., "Evaluating Unsupervised Text
Classification: Zero-shot and Similarity-based Approaches", NLPIR 2022.
Os CSVs em `data/raw/` são redistribuídos aqui sob os mesmos termos.

Efeito colateral: o reagrupamento resultou em 3 classes quase perfeitamente
balanceadas (~33/33/33%), eliminando o desbalanceamento 3,2x do dataset
original sem qualquer balanceamento artificial.

## Decisão arquitetural: híbrido batch + tempo real

Seguindo o princípio de que **arquiteturas de ML são pipelines, não
endpoints isolados**, o sistema combina dois modos de deploy:

- **Batch** (Airflow DAG, `dags/train_pipeline_dag.py`): ingestão do dataset
  completo → treino → salvamento do modelo. Roda offline, agendado.
- **Tempo real** (FastAPI, `src/api/main.py`): inferência síncrona por laudo
  individual, stateless, resposta em milissegundos.

Diferente de sistemas com busca vetorial (RAG, recomendação), não há aqui um
componente de reindexação — a atualização contínua se resume ao retreino
agendado via `schedule="@weekly"` na DAG.

**Nuvem (documentação teórica, Etapa 1):** entre AWS (SageMaker/Lambda/ECS),
GCP (Vertex AI/Cloud Run) e Azure (ML Studio/AKS), a recomendação é
**Google Cloud Run**: aceita o container Docker diretamente, escala a zero
(FinOps — sem custo parado), cold start baixo porque o artefato é leve
(TF-IDF + LogisticRegression, sem GPU), e não exige a complexidade de
rede/IAM de um cluster (ECS/AKS) para uma API única e stateless. Validação
rápida contra o framework de decisão da aula:

| Pergunta | Resposta |
|---|---|
| Custo por inferência | Baixo — CPU-only, milissegundos |
| Precisa do dataset completo? | Não — só o artefato treinado |
| Responde por evento? | Sim — `/predict` é stateless |
| Sempre carregado? | Não precisa — modelo leve, cold start rápido |
| CPU ou GPU? | CPU só |

O deploy real em nuvem é opcional/bônus; o obrigatório (stack local via
Docker Compose) está descrito abaixo.

## Resultados reais (medidos, não estimados)

Todos os números abaixo vieram de execuções reais neste ambiente de
desenvolvimento — nenhum é estimativa.

**Modelo baseline** (`src/training/train.py`, TF-IDF + LogisticRegression, seed=42):

| Métrica | Valor |
|---|---:|
| Acurácia (holdout de teste, 2.888 amostras) | 0,6195 |
| Macro F1 | 0,6158 |
| Tempo de treino | 3,64s |

**Latência da API — Etapa 1 baseline** (`scripts/bench_latency.py`, 200 requisições,
uvicorn local, sem container):

| Percentil | Latência |
|---|---:|
| p50 | 2,72 ms |
| p95 | 3,07 ms |
| p99 | 3,67 ms |

**Otimização ONNX — Etapa 4** (`scripts/export_onnx.py`, 300 chamadas, CPU):

| | sklearn (original) | ONNX Runtime | Speedup |
|---|---:|---:|---:|
| Inferência pura (sem overhead de API) | 0,673 ms | 0,087 ms | **7,70x** |

Saída ONNX validada byte-a-byte contra o pipeline sklearn (mesmo label,
mesmas probabilidades) — `models/model.onnx` já está no repositório.
Nota honesta: esse speedup é na camada de inferência pura; a latência
end-to-end da API (Etapa 1, ~2,6ms) já é dominada por overhead de
FastAPI/HTTP, não pelo modelo — o ganho visível ao trocar o backend de
inferência da API deve ser proporcionalmente menor que 5,44x. A comparação
completa end-to-end (API com sklearn vs. API com ONNX) é o entregável final
da Etapa 4.

## Estrutura do repositório

```
vitorium-mednlp-mlops/
├── .github/workflows/ci.yml       # lint (ruff) -> test (pytest) -> build (docker)
├── dags/train_pipeline_dag.py     # Airflow: ingest -> train -> save
├── data/{raw,processed}/          # CSVs (gitignored; baixar/gerar via make)
├── docker/
│   ├── docker-compose.yml         # api + prometheus + grafana
│   ├── prometheus.yml
│   └── grafana/provisioning/      # datasource + dashboard (5 paineis) pre-configurados
├── models/                        # model.joblib, model.onnx, metrics.json (versionados)
├── scripts/
│   ├── bench_latency.py           # benchmark real contra servidor rodando
│   └── export_onnx.py             # Etapa 4: exporta e valida ONNX
├── src/
│   ├── api/{main,schemas,metrics}.py
│   └── training/{preprocess,train}.py
├── tests/                         # pytest: smoke, schema (Pandera), integracao
├── Dockerfile
├── Makefile
├── pyproject.toml / poetry.lock    # deps (Poetry) + config ruff/pytest
├── requirements.txt                # espelho runtime p/ Docker (sem Poetry no container)
├── docs/SPEC.md / docs/VERSIONING.md
└── README.md
```

## Como rodar

```bash
make install                # poetry install (deps de producao + dev em grupos separados)
make train                  # baixa dados (preprocess.py baixa localmente) + treina
make test                   # pytest (7 testes)
poetry run uvicorn src.api.main:app --port 8000 &
make bench-latency          # latencia real contra o servidor local
make export-onnx            # Etapa 4: exporta e compara ONNX vs sklearn

make compose-up             # stack completa: API + Prometheus + Grafana
# Grafana: http://localhost:3000 (admin/admin) — dashboard "VITORIUM API" ja provisionado
# Prometheus: http://localhost:9090
```

## Status por etapa

- [x] **Etapa 1** — API FastAPI funcional, decisão arquitetural documentada,
      baseline de latência medido (2,57ms p50).
- [x] **Etapa 2 (parcial)** — DAG Airflow escrita e com sintaxe validada
      (TaskFlow API, 3 tasks); requer `airflow dags test` em ambiente com
      Airflow instalado para validação de execução real. Workflow GitHub
      Actions validado (lint → test → build com upload/download de
      artefato entre jobs).
- [x] **Etapa 3** — Docker Compose com API + Prometheus + Grafana, dashboard
      com 5 painéis provisionado automaticamente (total de requisições,
      latência p95, taxa de erro, RPS, predições por classe).
- [x] **Etapa 4 (parcial)** — ONNX exportado e validado (5,44x mais rápido
      na inferência pura); falta a comparação end-to-end via API e o vídeo
      STAR.

## Limitações e uso responsável

Este é um projeto acadêmico (FIAP Pós-Tech). O classificador não é validado
clinicamente, os rótulos de urgência são um proxy heurístico (ver seção
acima) e o sistema não deve ser usado para triagem médica real.
