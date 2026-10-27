#!/usr/bin/env bash
# Reproduz localmente a execucao real da DAG `vitorium_train_pipeline` com
# `airflow dags test` (tasks in-process, sqlite, sem scheduler/webserver).
#
# Uso:  scripts/run_airflow_dag_test.sh [data_logica]   (padrao: hoje, YYYY-MM-DD)
#
# Variaveis opcionais:
#   AIRFLOW_VENV  venv isolado do Airflow   (padrao: .venv-airflow na raiz do repo)
#   AIRFLOW_HOME  home descartavel          (padrao: .airflow-home na raiz do repo)
#   PYTHON_BIN    interpretador 3.11        (padrao: python3.11)
#
# Atencao: a task train_model regrava models/model.joblib e models/metrics.json.
# Para voltar aos artefatos versionados: git checkout -- models/
# O script nao apaga nada; so escreve no venv, no AIRFLOW_HOME e nas pastas
# que o proprio pipeline usa (data/processed/, models/).
set -euo pipefail

AIRFLOW_VERSION="2.10.5"
PY_MINOR="3.11"
CONSTRAINTS_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PY_MINOR}.txt"
# Pins do projeto (poetry.lock); instalados apos o Airflow, sem o arquivo de
# constraints, porque o Airflow core nao depende deles.
PROJECT_DEPS=("scikit-learn==1.9.1" "pandas==2.3.3" "joblib==1.6.0")

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_VENV="${AIRFLOW_VENV:-${PROJECT_ROOT}/.venv-airflow}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
LOGICAL_DATE="${1:-$(date +%F)}"

export AIRFLOW_HOME="${AIRFLOW_HOME:-${PROJECT_ROOT}/.airflow-home}"
export AIRFLOW__CORE__DAGS_FOLDER="${PROJECT_ROOT}/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
# Torna `import src...` resolvivel dentro das tasks.
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${AIRFLOW_VENV}/bin/airflow" ]]; then
    echo ">> Criando venv isolado em ${AIRFLOW_VENV} e instalando Airflow ${AIRFLOW_VERSION}"
    if command -v uv >/dev/null 2>&1; then
        uv venv "${AIRFLOW_VENV}" --python "${PYTHON_BIN}"
        uv pip install --python "${AIRFLOW_VENV}/bin/python" \
            "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINTS_URL}"
        uv pip install --python "${AIRFLOW_VENV}/bin/python" "${PROJECT_DEPS[@]}"
    else
        "${PYTHON_BIN}" -m venv "${AIRFLOW_VENV}"
        "${AIRFLOW_VENV}/bin/pip" install --upgrade pip
        "${AIRFLOW_VENV}/bin/pip" install \
            "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINTS_URL}"
        "${AIRFLOW_VENV}/bin/pip" install "${PROJECT_DEPS[@]}"
    fi
fi

AIRFLOW="${AIRFLOW_VENV}/bin/airflow"
mkdir -p "${AIRFLOW_HOME}"

echo ">> Airflow $("${AIRFLOW}" version) | AIRFLOW_HOME=${AIRFLOW_HOME}"
"${AIRFLOW}" db migrate
"${AIRFLOW}" dags list
"${AIRFLOW}" dags list-import-errors

echo ">> airflow dags test vitorium_train_pipeline ${LOGICAL_DATE}"
"${AIRFLOW}" dags test vitorium_train_pipeline "${LOGICAL_DATE}"

echo ">> Estados das tasks:"
"${AIRFLOW}" tasks states-for-dag-run vitorium_train_pipeline \
    "manual__${LOGICAL_DATE}T00:00:00+00:00"

echo ">> Concluido. models/ foi regravado; para restaurar: git checkout -- models/"
