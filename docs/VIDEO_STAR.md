# Roteiro do vídeo STAR — VITORIUM (até 5 minutos)

Roteiro para a gravação da entrega da Fase 3, no método STAR (Situation,
Task, Action, Result). O tempo total previsto é **4:45**, deixando 15
segundos de margem sobre o limite de 5 minutos. As falas foram escritas para
serem ditas em voz alta, em ritmo natural (cerca de 140 a 150 palavras por
minuto). Não é preciso decorar palavra por palavra; o importante é manter os
números exatamente como estão.

## 1. Linha do tempo

| Bloco | Início | Fim | Duração | Tela principal |
|---|---|---|---|---|
| S — Situação | 0:00 | 0:30 | 0:30 | README (capa e nota de honestidade) |
| T — Tarefa | 0:30 | 0:55 | 0:25 | README (status por etapa) |
| A1 — Repositório e CI/CD | 0:55 | 1:25 | 0:30 | GitHub: código e run verde do Actions |
| A2 — Container e API | 1:25 | 2:05 | 0:40 | Terminal com `docker compose`, Swagger `/docs` |
| A3 — Monitoramento | 2:05 | 2:45 | 0:40 | Prometheus Alerts, Grafana |
| A4 — Retreino e registro de imagem | 2:45 | 3:15 | 0:30 | Evidência do Airflow, pacote no GHCR |
| A5 — Otimização ONNX | 3:15 | 3:40 | 0:25 | `models/api_latency_comparison.json` |
| R — Resultados | 3:40 | 4:25 | 0:45 | README (resultados) e JSONs de métricas |
| Encerramento | 4:25 | 4:45 | 0:20 | `docs/MODEL_CARD.md` |

## 2. Roteiro por bloco

### S — Situação (0:00 a 0:30)

**Mostrar:** README do repositório no GitHub, rolando da capa até a nota de
honestidade sobre os rótulos.

**Falar:**

> Imagine um hospital que recebe centenas de laudos por dia e precisa saber
> quais olhar primeiro. Eu sou o Vitor, e nesta Fase 3 construí o VITORIUM,
> um classificador leve de texto médico que separa textos em urgente,
> atenção e normal. Já deixo claro: esses rótulos são um mapeamento
> heurístico de categorias de doença de um dataset público de abstracts em
> inglês. É um projeto de MLOps, não uma ferramenta clínica.

### T — Tarefa (0:30 a 0:55)

**Mostrar:** seção "Status por etapa" do README.

**Falar:**

> A tarefa não era ter o modelo mais sofisticado, e sim a esteira em volta
> dele: API em container, CI/CD com lint, teste e build, retreino orquestrado
> no Airflow, monitoramento com Prometheus e Grafana, e uma otimização de
> latência medida, não estimada. Com uma regra: nenhum número sem um arquivo
> de origem no repositório.

### A1 — Repositório e CI/CD (0:55 a 1:25)

**Mostrar:** `src/training/train.py` (função `build_pipeline`) por 5
segundos; depois a aba Actions com o último run verde, abrindo o grafo
`lint → test → build`.

**Falar:**

> O modelo é TF-IDF com regressão logística, em scikit-learn, com seed fixa.
> A cada push na main, o GitHub Actions roda três jobs encadeados: lint com
> Ruff, testes com Pytest, incluindo validação de schema com Pandera, e o
> build da imagem Docker, que é publicada no GitHub Container Registry com a
> tag do commit.

### A2 — Container e API (1:25 a 2:05)

**Mostrar:**

1. Terminal: `docker compose -f docker/docker-compose.yml ps` com os três
   serviços de pé e a API `healthy` (o `up --build` já foi feito antes da
   gravação; ver checklist).
2. Navegador em <http://localhost:8000/docs>: `POST /predict` → *Try it out*
   → *Execute* com o exemplo pré-preenchido (dor torácica e troponina).
   Destacar `urgencia`, `probabilidades` e `latencia_ms` na resposta.
3. `GET /health` → *Execute*, mostrando `model_loaded: true`.

**Falar:**

> Com um comando, o Docker Compose sobe a API, o Prometheus e o Grafana, e os
> três estão saudáveis. No Swagger, envio um texto clínico em inglês para o
> predict. A resposta traz a classe, as probabilidades das três classes e o
> tempo de inferência. E o health confirma que o modelo está carregado.

### A3 — Monitoramento (2:05 a 2:45)

**Mostrar:**

1. <http://localhost:9090/alerts>: as quatro regras (`VitoriumApiDown`,
   `VitoriumHighErrorRate`, `VitoriumHighLatencyP95`,
   `VitoriumPredictionDistributionShift`).
2. `docs/MONITORING.md` no GitHub, clicando em um link de playbook da tabela
   de alertas por 3 segundos.
3. <http://localhost:3000>, dashboard "VITORIUM API", janela "Last 15
   minutes", com os cinco painéis populados.

**Falar:**

> A API exporta três métricas para o Prometheus: contagem de requisições,
> histograma de latência e predições por classe. Aqui estão quatro alertas:
> API fora do ar, erro acima de cinco por cento, p95 acima de cinquenta
> milissegundos e mudança na distribuição das predições, que é o meu proxy
> de drift, porque em produção não existe rótulo verdadeiro. Cada alerta tem
> um playbook. E no Grafana, os cinco painéis com o tráfego que acabei de
> gerar.

### A4 — Retreino e registro de imagem (2:45 a 3:15)

**Mostrar:** `dags/train_pipeline_dag.py` (decorador `@dag` com
`schedule="@weekly"` e as quatro tasks), depois o log de execução da DAG em
`docs/evidence/`, depois a página do pacote
`ghcr.io/vdfs89/vitorium-mednlp-mlops` com as tags `latest` e por commit.

**Falar:**

> O retreino é uma DAG do Airflow com quatro tarefas, ingestão, treino,
> exportação do ONNX com checagem de equivalência, e relatório, agendada
> semanalmente. Aqui está o log da execução real, quatro de quatro com
> sucesso. E esta é a imagem publicada no GHCR,
> com uma tag por commit, que também é o meu ponto de rollback.

### A5 — Otimização ONNX (3:15 a 3:40)

**Mostrar:** `models/onnx_latency_comparison.json` e, em seguida,
`models/api_latency_comparison.json`, aberto no GitHub ou no editor.

**Falar:**

> Para a otimização, exportei o pipeline para ONNX e conferi que os rótulos
> batem com os do scikit-learn. A API pode servir com qualquer um dos dois
> backends, escolhido por variável de ambiente, e medi a latência de ponta a
> ponta nos dois casos.

### R — Resultados (3:40 a 4:25)

**Mostrar:** `models/metrics.json` (acurácia, macro F1 e o bloco da classe
`normal`) e de novo `models/api_latency_comparison.json`.

**Falar:**

> Os resultados, todos medidos e com arquivo de origem. Acurácia de 61,9 por
> cento e F1 macro de 0,616 em 2.888 textos de teste; a classe normal é a mais
> fraca, F1 de 0,46, por ser genérica. Na inferência pura, o ONNX é cerca de 4
> vezes mais rápido, com os mesmos rótulos em todo o teste. Na API, o p50 caiu
> de 3,4 para 2,5 milissegundos, 1,35 vez, porque ali o tempo é dominado por
> HTTP. E uma nota de integridade: a primeira exportação ONNX marcava 7,7
> vezes, mas não era equivalente ao modelo; corrigi e publiquei o número
> real.

### Encerramento (4:25 a 4:45)

**Mostrar:** `docs/MODEL_CARD.md`, seção "Fora de escopo".

**Falar:**

> As limitações estão no Model Card: rótulos heurísticos e corpus só em
> inglês, por isso o VITORIUM não deve ser usado para decisão clínica. A
> arquitetura de nuvem recomendada, Cloud Run, está documentada no README.
> Obrigado.

## 3. Fonte de cada número dito no vídeo

| Número falado | Valor no arquivo | Arquivo |
|---|---|---|
| 61,9% de acurácia | `accuracy`: 0,61946 | `models/metrics.json` |
| F1 macro 0,616 | `macro_f1`: 0,61576 | `models/metrics.json` |
| 2.888 textos de teste | `test_rows`: 2888 | `models/metrics.json` |
| F1 0,46 da classe `normal` | `normal.f1-score`: 0,46272 | `models/metrics.json` |
| ~4x mais rápido (inferência pura) | `onnx_mean_ms` 0,2653; `sklearn_mean_ms` 1,0841; `speedup_x` 4,09 | `models/onnx_latency_comparison.json` |
| mesmos rótulos nos 2.888 textos | `parity.label_agreement` 1,0; `parity.n_texts` 2888 | `models/onnx_latency_comparison.json` |
| p50 de 3,4 para 2,5 ms, 1,35x | `sklearn.p50_ms` 3,401; `onnx.p50_ms` 2,514; `p50_x` 1,353 | `models/api_latency_comparison.json` |
| 7,7x da exportação antiga | errata documentada | `docs/MODEL_CARD.md`, seção 5 |
| Limiares 5% e 50 ms | regras de alerta | `docker/alerts.yml` |

Use no vídeo somente os valores desta tabela; todos vêm dos arquivos JSON
versionados.

## 4. Checklist antes de gravar

1. **CI verde.** Abrir a aba Actions e confirmar que o último run da `main`
   está verde nos três jobs. Pelo terminal: `gh run list --branch main --limit 3`.
2. **Números preenchidos.** Confirmar que `models/api_latency_comparison.json`
   existe e que este roteiro não tem mais nenhum marcador entre chaves duplas.
3. **Subir a stack** (na raiz do repositório):

   ```bash
   docker compose -f docker/docker-compose.yml up --build -d
   docker compose -f docker/docker-compose.yml ps
   curl.exe http://localhost:8000/health
   ```

   Esperar a API aparecer como `healthy` (o healthcheck roda a cada 30 s).
4. **Gerar tráfego** de 2 a 3 minutos antes de gravar, com textos variados
   para não concentrar todas as predições em uma única classe (PowerShell):

   ```powershell
   $textos = @(
     "Patient presents with acute chest pain radiating to the left arm, elevated troponin levels and ECG changes consistent with myocardial infarction.",
     "Biopsy revealed adenocarcinoma of the colon with hepatic metastases; adjuvant chemotherapy was initiated after resection.",
     "Sudden onset of right-sided weakness and aphasia; CT showed an ischemic stroke in the left middle cerebral artery territory.",
     "We review general mechanisms of inflammation, fever and tissue repair in chronic pathological conditions.",
     "Endoscopy showed a gastric ulcer with signs of recent bleeding; proton pump inhibitor therapy was started."
   )
   for ($i = 0; $i -lt 300; $i++) {
     $body = @{ texto = $textos[$i % $textos.Count] } | ConvertTo-Json
     Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body | Out-Null
     Start-Sleep -Milliseconds 300
   }
   ```

   Se o alerta `VitoriumPredictionDistributionShift` aparecer como
   *pending*, isso é esperado com tráfego sintético. Diga isso em uma frase
   em vez de esconder.
5. **Abrir as abas na ordem do roteiro:** README no GitHub, Actions (run
   verde), `src/training/train.py`, terminal, <http://localhost:8000/docs>,
   <http://localhost:9090/alerts>, `docs/MONITORING.md`,
   <http://localhost:3000> (dashboard "VITORIUM API", "Last 15 minutes"),
   `dags/train_pipeline_dag.py`, evidência do Airflow em `docs/evidence/`,
   pacote no GHCR, `models/onnx_latency_comparison.json`,
   `models/api_latency_comparison.json`, `models/metrics.json`,
   `docs/MODEL_CARD.md`.
6. **Gravação:** resolução 1080p, zoom do navegador em 125% para o código
   ficar legível, notificações desligadas, microfone testado.
7. **Ensaiar uma vez com cronômetro.** Se passar de 4:45, corte primeiro o
   item 2 do bloco A3 (link do playbook) e o código da DAG no bloco A4.

## 5. O que NÃO dizer

- Que o modelo faz triagem clínica, está "validado", "pode ser usado em
  hospital" ou "ajuda médicos a decidir". Os rótulos são heurísticos.
- Que o sistema está implantado na nuvem. O que existe é a arquitetura de
  nuvem documentada (Cloud Run, no README) e a imagem pública no GHCR. A
  execução demonstrada é local, com Docker Compose.
- Que o modelo funciona com laudos em português. O corpus é 100% em inglês;
  texto em português cai quase sempre em `normal`.
- Que o ONNX deixou "a API 4 vezes mais rápida". O ~4x é só da inferência
  pura; na API o ganho de p50 é 1,35x (`models/api_latency_comparison.json`).
  E nunca o 7,7x antigo, que veio de uma exportação não equivalente.
- Que o modelo tem "alta acurácia" ou é "muito preciso". 61,9% em três
  classes balanceadas é um baseline honesto, não um resultado forte.
- Que o monitoramento mede acurácia em produção ou "detecta drift". Ele
  acompanha a distribuição das predições como proxy.
- Que os alertas notificam por e-mail ou Slack. Não há Alertmanager
  configurado; os alertas aparecem na interface do Prometheus.
- Que o retreino semanal corrige drift. Ele retreina sobre o mesmo corpus
  público; só corrige algo quando entram dados novos rotulados.
- Qualquer número que não esteja na tabela da seção 3.
