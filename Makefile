.PHONY: install install-pip lint type test quality download-index build-index run eval docker-build docker-run docker-compose-up

install:
	poetry install --extras "dev eval tracking"

install-pip:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev,eval,tracking]"

lint:
	poetry run ruff check .

type:
	poetry run mypy src scripts app

test:
	poetry run pytest -q

quality: lint type test

download-index:
	poetry run python scripts/download_wikipedia_subset.py --config configs/default.yaml

build-index:
	poetry run python scripts/build_index.py --config configs/default.yaml

run:
	poetry run streamlit run app/streamlit_app.py

eval:
	poetry run python scripts/evaluate_ragas.py --config configs/default.yaml

docker-build:
	docker build -t self-rag-wikipedia-demo .

docker-run:
	docker run --rm -p 8501:8501 -v "$$(pwd)/data:/app/data" self-rag-wikipedia-demo

docker-compose-up:
	docker compose up --build
