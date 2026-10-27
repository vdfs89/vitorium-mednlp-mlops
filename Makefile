ENV_FILE := $(if $(wildcard .env),--env-file .env,)
COMPOSE := docker compose $(ENV_FILE) -f docker/docker-compose.yml

.PHONY: help install preprocess train test lint export-onnx bench-latency compare-backends \
	airflow-test test-alerts docker-build docker-run compose-up compose-down

help: ## Lista os alvos disponiveis
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

install: ## Instala deps de producao + dev (Poetry)
	poetry install --no-root

preprocess: ## Gera data/processed/ a partir de data/raw/
	PYTHONPATH=. poetry run python -m src.training.preprocess

train: preprocess ## Treina o pipeline TF-IDF + LogisticRegression
	PYTHONPATH=. poetry run python -m src.training.train

export-onnx: ## Exporta model.onnx, valida paridade e mede inferencia pura
	PYTHONPATH=. poetry run python scripts/export_onnx.py

test: ## Roda a suite pytest
	PYTHONPATH=. poetry run pytest tests/ -v

lint: ## Ruff
	poetry run ruff check .

bench-latency: ## Latencia de uma API ja rodando em :8000
	PYTHONPATH=. poetry run python scripts/bench_latency.py --url http://127.0.0.1:8000 --n 200

compare-backends: ## Latencia end-to-end sklearn vs ONNX (sobe 2 servidores)
	PYTHONPATH=. poetry run python scripts/compare_backends.py --rounds 5 --n 300 --warmup 50

airflow-test: ## Executa a DAG de verdade (airflow dags test) e restaura models/
	bash scripts/run_airflow_dag_test.sh && git checkout -- models/

test-alerts: ## Testes unitarios das regras de alerta (promtool via Docker)
	docker run --rm -v "$(CURDIR)/docker:/w" -w /w --entrypoint promtool prom/prometheus:v2.55.1 test rules alerts_test.yml

docker-build: ## Build da imagem da API
	docker build -t vitorium-api:latest .

docker-run: docker-build ## Roda a API isolada em :8000
	docker run --rm -p 8000:8000 vitorium-api:latest

compose-up: ## Sobe API + Prometheus + Grafana
	$(COMPOSE) up --build -d

compose-down: ## Derruba a stack
	$(COMPOSE) down
