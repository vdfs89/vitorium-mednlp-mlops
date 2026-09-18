.PHONY: install preprocess train test lint bench-latency export-onnx docker-build docker-run compose-up compose-down

install:
	poetry install --no-root

preprocess:
	PYTHONPATH=. poetry run python -m src.training.preprocess

train: preprocess
	PYTHONPATH=. poetry run python -m src.training.train

test:
	PYTHONPATH=. poetry run pytest tests/ -v

lint:
	poetry run ruff check .

bench-latency:
	PYTHONPATH=. poetry run python scripts/bench_latency.py --url http://127.0.0.1:8000 --n 200

export-onnx:
	PYTHONPATH=. poetry run python scripts/export_onnx.py

docker-build:
	docker build -t vitorium-api:latest .

docker-run: docker-build
	docker run --rm -p 8000:8000 vitorium-api:latest

compose-up:
	docker compose -f docker/docker-compose.yml up --build -d

compose-down:
	docker compose -f docker/docker-compose.yml down