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

| Requisito (enunciado Fase 3) | Onde está | Status |
|---|---|---|
| Pipeline CI/CD com GitHub Actions (lint → test → build) | `.github/workflows/ci.yml` | Sintaxe validada; 3 jobs encadeados com `needs`, artefato do modelo passado entre jobs |
| DAG/script Airflow de treino/retreino | `dags/train_pipeline_dag.py` | Sintaxe validada (TaskFlow API); requer `airflow dags test` em ambiente com Airflow para validação de execução |
| Dockerfile funcional da API | `Dockerfile` | Escrito e revisado (non-root, healthcheck); build não executado neste ambiente (sem docker daemon) — validar com `make docker-build` |
| Docker Compose: API + Prometheus + Grafana | `docker/docker-compose.yml` + `docker/prometheus.yml` + `docker/grafana/provisioning/` | YAML/JSON validados; dashboard com 5 painéis pré-provisionado |
| Histórico de commits organizado | commits deste repo a partir de agora | Primeiro commit é o scaffold; commits seguintes devem ser um por etapa/funcionalidade, não um monólito final |
| Scikit-Learn (modelo leve) | `src/training/train.py` (TF-IDF + LogisticRegression) | Treinado e avaliado: acc 0,6195 / macro F1 0,6158 (real, ver `models/metrics.json`) |
| FastAPI (`/predict`, `/health`) | `src/api/main.py` | Testado (7 testes em `tests/`), rodando ao vivo neste ambiente |
| `prometheus_client` instrumentando latência + contagem | `src/api/metrics.py` | Middleware real, testado (`test_metrics_endpoint_exposes_prometheus_format`) |
| Airflow orquestrando treino | `dags/train_pipeline_dag.py` | Ver acima |
| Otimização de latência (ONNX/quantização/pruning) | `scripts/export_onnx.py` | Executado: ONNX 5,44x mais rápido que sklearn na inferência pura, saída validada byte-a-byte |
| Comparação de latência original vs. otimizado | `models/onnx_latency_comparison.json` + `scripts/bench_latency.py` | Parcial: comparação de inferência pura pronta; falta rodar `bench_latency.py` contra a API já servindo via ONNX (troca de backend em `src/api/main.py`) |
| Decisão de arquitetura de nuvem documentada no README | `README.md` (seção "Decisão arquitetural") | Feito: Cloud Run, com justificativa e checklist da aula |
| Vídeo STAR (≤ 5 min) | — | Pendente — depende do pipeline completo (Etapas 2-4) estar rodando de ponta a ponta |

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
mais rápida de treinar (3,64s vs. RF tipicamente mais lento em texto de alta
dimensionalidade) e por ter conversão ONNX mais direta e sem ambiguidade de
`zipmap`. Trade-off documentado, não uma omissão.

## 4. Convenções de código (clean code)

- Toda função pública tem docstring explicando o *porquê*, não só o *o quê*.
- `from __future__ import annotations` + type hints em todo módulo novo.
- Nenhum número em README/vídeo sem uma fonte rastreável em código
  (`models/*.json`) — ver padrão de integridade de métricas já usado nos
  outros projetos do portfólio.
- `ruff check .` limpo antes de qualquer commit (rodar `make lint`).
- Testes não são decorativos: cada teste em `tests/` falha de verdade se o
  comportamento quebrar (rodados e confirmados neste ambiente, 7/7 passando).
- Um commit por unidade de trabalho coerente (uma etapa, uma correção, uma
  feature) — não empilhar Etapas 2-4 num commit só.

## 5. Próximos passos (na ordem)

1. Validar `docker build` / `docker compose up` num ambiente com Docker
   daemon (este ambiente de geração não tem um disponível).
2. Rodar `airflow dags test vitorium_train_pipeline 2026-01-01` num ambiente
   com Airflow instalado.
3. Trocar o backend de inferência da API para usar `models/model.onnx` via
   `onnxruntime` e re-rodar `bench_latency.py` para a comparação end-to-end
   (fecha a Etapa 4).
4. Subir a stack completa via `make compose-up`, gerar tráfego real e
   printar/exportar o dashboard do Grafana (evidência exigida da Etapa 3).
5. Roteirizar e gravar o vídeo STAR com os números reais deste documento.
