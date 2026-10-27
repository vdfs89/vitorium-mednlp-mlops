# VITORIUM — Especificação Técnica (SDD)

Este documento existe para o projeto não se perder ao longo das 4 etapas:
toda decisão de arquitetura e todo requisito obrigatório do desafio tem um
lugar único e rastreável aqui. Mudanças de escopo passam por aqui antes de
virar código.

## 1. Problema e escopo

Classificador de texto (NLP) leve para triagem de laudos médicos, servido
via API REST, com ciclo de vida completo em MLOps: CI/CD, orquestração de
retreino, monitoramento e otimização de latência. Ver `README.md` para o
mapeamento de dataset/rótulos e a nota de honestidade sobre a proxy de
urgência.

**Fora de escopo, deliberadamente:** sofisticação de modelo (deep learning,
RAG, ensembles), deploy real em nuvem paga, autenticação/autorização na API.
O que é avaliado é a esteira MLOps, não a complexidade do modelo — manter o
modelo simples é uma decisão de design, não uma limitação.

## 2. Rastreabilidade de requisitos obrigatórios

Status atualizado em 27/10/2026. "Verificado" significa executado de verdade,
com a evidência citada; nada aqui é só "escrito".

| Requisito (enunciado Fase 3) | Onde está | Status e evidência |
|---|---|---|
| Pipeline CI/CD com GitHub Actions (lint → test → build) | `.github/workflows/ci.yml` | **Verificado** no run de `57cb177` (lint, test, build + push no GHCR). Adicionados depois e simulados localmente, a confirmar no próximo run: export do ONNX no `test` e smoke test do container (`/health` com `backend: onnx`) antes do push |
| DAG/script Airflow de treino/retreino | `dags/train_pipeline_dag.py` | **Verificado**: `airflow dags test` com Airflow 2.10.5, 4/4 tasks `success` (`ingest_data → train_model → export_onnx → save_model_report`), métricas idênticas às versionadas. Evidência: `docs/evidence/airflow_dags_test.log`, `airflow_task_states.txt`; reprodução: `make airflow-test` |
| Dockerfile funcional da API | `Dockerfile` (multi-stage) + `.dockerignore` | **Verificado**: imagem multi-stage construída com `docker compose up --build` na máquina do Vitor (commit `31da732`), 3 containers no ar e `/health` = `{"status":"ok","model_loaded":true,"backend":"onnx"}` — ONNX carregando dentro do `python:3.11-slim` (`docs/evidence/docker_compose_onnx.txt`). Lint `hadolint` sem achados (`docs/evidence/promtool_hadolint.txt`) |
| Docker Compose: API + Prometheus + Grafana | `docker/docker-compose.yml`, `docker/prometheus.yml`, `docker/grafana/provisioning/` | **Verificado end-to-end**: 3 containers up, target UP, dashboard de 5 painéis com tráfego real. Evidência: `docs/evidence/*.png` |
| Monitoramento com alertas | `docker/alerts.yml` + `docs/MONITORING.md` | **Verificado**: `promtool check rules` (9 regras) e `promtool test rules` passando (`docs/evidence/promtool_hadolint.txt`); as 4 regras carregadas no Prometheus real da stack, todas inativas sem incidente (`docs/evidence/prometheus-alerts-loaded.png`); playbook por alerta |
| Histórico de commits organizado | `git log` | Commits semânticos (Conventional Commits) a partir da `v0.1.1`, um por unidade de trabalho; os 3 primeiros commits são o scaffold inicial |
| Scikit-Learn (modelo leve) | `src/training/train.py` | **Verificado**: acc 0,6195 / macro F1 0,6158 (`models/metrics.json`), determinístico (re-treino no Airflow e local reproduz os mesmos valores) |
| FastAPI (`/predict`, `/health`) | `src/api/main.py`, `src/api/backends.py` | **Verificado**: 15 testes (`tests/`), incluindo integração, paridade entre backends e `/health` 503 sem modelo |
| `prometheus_client` instrumentando latência + contagem | `src/api/metrics.py` | **Verificado**: middleware real, testado; inclui `vitorium_model_info{backend}` |
| Otimização de latência (ONNX/quantização/pruning) | `src/training/onnx_export.py`, `scripts/export_onnx.py`, `src/api/backends.py` | **Verificado**: API serve ONNX por padrão; paridade com sklearn em 100% dos 2.888 textos de teste (máx. dif. 1,6e-7); inferência pura 4,09x mais rápida (`models/onnx_latency_comparison.json`) |
| Comparação de latência original vs. otimizado | `scripts/compare_backends.py` → `models/api_latency_comparison.json` | **Verificado**: end-to-end via HTTP, 5 rodadas intercaladas × 300 req: p50 3,401 → 2,514 ms (1,35x), p95 4,486 → 3,115 ms |
| Decisão de arquitetura de nuvem documentada no README | `README.md` ("Decisão arquitetural") + ADR-002 | Feito: Cloud Run, com o checklist da aula; imagem pública no GHCR como artefato de deploy. Deploy real não feito (bônus) |
| Model Card / limitações | `docs/MODEL_CARD.md` | Feito: métricas por classe, vieses, cenários de falha, errata do ONNX |
| Vídeo STAR (≤ 5 min) | `docs/VIDEO_STAR.md` | Roteiro de 4:45 (≈ 3,8 min de fala, cada número com arquivo-fonte); **gravação pendente** |

## 3. Decisões de arquitetura (ADR resumido)

**ADR-001 — Híbrido batch + tempo real.** Airflow (batch) treina; FastAPI
(tempo real) serve. Sem componente de reindexação vetorial — não há busca
por embeddings neste projeto, diferente de RAG/recomendação.

**ADR-002 — Cloud Run como alvo teórico de deploy.** Modelo leve, CPU-only,
stateless → serverless com scale-to-zero é o ajuste mais direto entre as
opções do curso (AWS SageMaker/Lambda/ECS, GCP Vertex AI/Cloud Run, Azure ML
Studio/AKS). Deploy real é bônus, não bloqueante.

**ADR-003 — Mapeamento heurístico de urgência.** Ver `README.md` e o
docstring de `src/training/preprocess.py`. Decisão motivada por honestidade
de métricas: preferimos um proxy explícito e documentado a inventar rótulos
de triagem clínica que o dataset não tem.

**ADR-004 — TF-IDF + LogisticRegression, não Random Forest.** O enunciado
sugere "TF-IDF + Random Forest ou modelo leve similar" como exemplo, não
como exigência. LogisticRegression foi escolhida por ser igualmente leve,
mais rápida de treinar (3,79 s, `models/metrics.json`; RF tipicamente é mais lento em texto de alta
dimensionalidade) e por ter conversão ONNX mais direta. Trade-off documentado,
não uma omissão.

**ADR-005 — Exportador ONNX próprio, com paridade obrigatória.** A
conversão padrão do skl2onnx não reproduzia o `TfidfVectorizer` com
`stop_words="english"` (35 de 2.888 rótulos divergentes, `docs/evidence/onnx_stock_export_parity.json`) e dependia do
locale `en_US.UTF-8`. `src/training/onnx_export.py` monta o grafo
equivalente e `scripts/export_onnx.py` aborta se qualquer texto do teste
divergir além de 1e-5. A mesma checagem roda como task da DAG, então um
retreino nunca publica um ONNX não equivalente.

## 4. Convenções de código (clean code)

- Toda função pública tem docstring explicando o *porquê*, não só o *o quê*.
- `from __future__ import annotations` + type hints em todo módulo novo.
- Nenhum número em README/vídeo sem uma fonte rastreável em código
  (`models/*.json`) — ver padrão de integridade de métricas já usado nos
  outros projetos do portfólio.
- `ruff check .` limpo antes de qualquer commit (rodar `make lint`).
- Testes não são decorativos: cada teste em `tests/` falha de verdade se o
  comportamento quebrar (15 testes, rodados localmente e no CI).
- Um commit por unidade de trabalho coerente (uma etapa, uma correção, uma
  feature) — não empilhar Etapas 2-4 num commit só.

## 5. Próximos passos

1. Gravar o vídeo STAR seguindo `docs/VIDEO_STAR.md` e taguear `v1.0.0`.
2. (Bônus) `gcloud run deploy --image ghcr.io/vdfs89/vitorium-mednlp-mlops:<sha>`.
3. (Produção) Alertmanager com roteamento, e amostragem rotulada periódica
   para medir acurácia real (hoje o drift é monitorado por proxy).
