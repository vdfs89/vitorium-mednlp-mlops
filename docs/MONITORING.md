# Plano de monitoramento em produção — VITORIUM

Este documento descreve o que é medido, quais objetivos de serviço (SLOs)
propomos, como ler o dashboard, quais alertas existem e o que fazer quando
cada um dispara. Ele também deixa explícito o que **não** está implementado,
para que ninguém confunda proposta com realidade.

Componentes envolvidos (todos neste repositório):

| Componente | Arquivo | Papel |
|---|---|---|
| Instrumentação da API | `src/api/metrics.py` | Middleware que alimenta as métricas e expõe `/metrics` |
| Prometheus | `docker/prometheus.yml` | Coleta `api:8000/metrics` a cada 5 s (`scrape_interval: 5s`) |
| Regras de alerta | `docker/alerts.yml` | Expressões canônicas dos 4 alertas desta página |
| Grafana | `docker/grafana/provisioning/` | Datasource e dashboard "VITORIUM API" (5 painéis) provisionados |
| Stack local | `docker/docker-compose.yml` | Sobe `api`, `prometheus` e `grafana` |

> **Roteamento de alertas.** Não há Alertmanager configurado neste
> repositório. Os alertas são avaliados pelo Prometheus e ficam visíveis em
> <http://localhost:9090/alerts> (estados *inactive*, *pending* e *firing*),
> mas nenhuma notificação (e-mail, Slack, pager) é enviada. Em uma
> implantação real, o próximo passo seria adicionar o Alertmanager com rotas
> por severidade.

## 1. O que é medido

### 1.1 Métricas técnicas (expostas pela API)

| Métrica | Tipo | Labels | O que representa |
|---|---|---|---|
| `vitorium_requests_total` | Counter | `method`, `endpoint`, `status_code` | Requisições atendidas, por rota e código HTTP (inclusive 500 de exceção não tratada) |
| `vitorium_request_latency_seconds` | Histogram | `method`, `endpoint` | Latência medida no middleware (inclui validação, inferência e serialização; não inclui rede do cliente) |
| `vitorium_predictions_total` | Counter | `urgencia` | Predições emitidas por classe (`normal`, `atencao`, `urgente`) |
| `up{job="vitorium-api"}` | Gauge (Prometheus) | — | 1 se o último scrape funcionou, 0 caso contrário |

Os buckets do histograma são 5, 10, 25, 50, 100, 250, 500 ms e 1; 2,5; 5 s
(`src/api/metrics.py`). Consequência prática: o limiar de 50 ms do SLO de
latência coincide com uma borda de bucket, então a comparação "p95 acima ou
abaixo de 50 ms" é precisa. Já valores pequenos (a API mede p95 de 3,07 ms
localmente, `models/latency_baseline.json`) caem todos no primeiro bucket e
o `histogram_quantile` apenas interpola dentro de 0 a 5 ms. Para números
exatos de latência baixa, a fonte é o benchmark (`scripts/bench_latency.py`),
não o Prometheus.

**Lacunas conhecidas da instrumentação:**

- Exceções não tratadas são contadas: o middleware registra contagem e
  latência num bloco `finally`, como status 500, e relança a exceção para o
  Starlette responder e logar (teste:
  `tests/test_hardening.py::test_unhandled_inference_error_is_counted_once_as_500`).
  O stack trace continua só nos logs do container.
- O label `endpoint` é o **template da rota** (`/predict`, `/health`,
  `/metrics`), nunca a URL crua; caminhos desconhecidos viram um único
  `endpoint="unmatched"`, então varreduras de URL não criam séries novas
  (teste: `test_unknown_paths_share_a_single_metric_label`).
- Respostas 4xx (por exemplo 422 de validação) são contadas, mas não entram
  na taxa de erro, por definição.
- `/health` e `/metrics` também passam pelo middleware. Toda consulta de
  latência e de taxa de erro do modelo deve filtrar `endpoint="/predict"`.

### 1.2 Proxies de qualidade do modelo

Em produção não existe rótulo verdadeiro: ninguém informa à API qual era a
urgência correta. Portanto **não é possível medir acurácia ou F1 em tempo
real**. O que medimos é:

1. **Distribuição das predições** (`vitorium_predictions_total`) comparada
   com o prior de treino: `urgente` 0,345, `atencao` 0,323, `normal` 0,333
   (`README.md`, tabela de mapeamento). Mudanças grandes indicam que a
   entrada mudou (deriva de dados), que o tráfego é atípico, ou que o modelo
   quebrou.
2. **Revisão periódica de amostra rotulada** (proposta, não implementada):
   mensalmente, sortear predições registradas, pedir rótulo a um revisor e
   calcular F1 macro e recall de `urgente` sobre a amostra. É a única forma
   de medir qualidade real; a distribuição de predições só sinaliza que vale
   a pena olhar.

Referência importante: mesmo no conjunto de teste, o modelo não reproduz o
prior exatamente. Pelas métricas de `models/metrics.json`, a fração prevista
no teste é 0,347 `atencao`, 0,299 `normal` e 0,354 `urgente` (conta em
`docs/MODEL_CARD.md`, seção 4). O desvio máximo em relação ao prior é de
cerca de 0,034, então o limiar de 0,15 do alerta não dispara pelo viés normal
do modelo, só por mudança real de tráfego.

**Proxies propostos para uma próxima versão** (não implementados): histograma
do comprimento do texto de entrada, fração de tokens fora do vocabulário
TF-IDF por requisição e histograma da `confianca`. Os três detectariam os
cenários de falha mais perigosos descritos em `docs/MODEL_CARD.md` (texto em
português, texto vazio), que hoje aparecem apenas como excesso de `normal`.

## 2. SLOs propostos

| SLO | Objetivo | Janela | Como medir (PromQL) |
|---|---|---|---|
| Disponibilidade | 99,5% | 30 dias | `avg_over_time(up{job="vitorium-api"}[30d])` |
| Latência `/predict` | p95 < 50 ms | contínuo (avaliado em 5 min) | `histogram_quantile(0.95, sum by (le) (rate(vitorium_request_latency_seconds_bucket{endpoint="/predict"}[5m])))` |
| Taxa de erro | 5xx < 1% das requisições | 30 dias | `sum(increase(vitorium_requests_total{status_code=~"5.."}[30d])) / sum(increase(vitorium_requests_total[30d]))` |

**Orçamento de erro de disponibilidade:** 0,5% de 30 dias = 216 minutos
(3 h 36 min) de indisponibilidade por mês.

**Por que 50 ms se a medida local é 3,07 ms p95?** O baseline
(`models/latency_baseline.json`) foi medido sem container, sem rede e sem
concorrência. O SLO deixa folga para container, rede de nuvem, cold start e
picos de carga, e continua bem abaixo do que um sistema de apoio à triagem
precisaria.

**Limitações honestas da medição:**

- O Prometheus do Compose não tem volume persistente nem retenção
  configurada (`docker/docker-compose.yml`), então usa o padrão de 15 dias e
  perde o histórico quando o container é recriado. As consultas de 30 dias
  acima são a definição correta, mas nesta stack local só têm dados de
  quanto tempo o container estiver de pé. Em produção: volume persistente e
  `--storage.tsdb.retention.time=35d` (ou armazenamento remoto).
- O SLO de erro (1%) é mais estrito que o alerta de erro (5% por 5 min). O
  alerta pega incidentes agudos; o consumo lento do orçamento deve ser
  acompanhado no dashboard. Alertas de *burn rate* em múltiplas janelas
  seriam a evolução natural.

## 3. Dashboard Grafana "VITORIUM API"

Provisionado de `docker/grafana/provisioning/dashboards/vitorium-api.json`,
acessível em <http://localhost:3000> (usuário/senha de `GF_SECURITY_ADMIN_USER`/`GF_SECURITY_ADMIN_PASSWORD` do `.env`; sem `.env`, admin/admin; ou anônimo como Viewer).
Evidência com tráfego real: `docs/evidence/grafana-dashboard-populated.png`.

| Painel | Consulta | Como ler |
|---|---|---|
| Total de Requisicoes | `sum(vitorium_requests_total)` | Volume acumulado desde o último start da API (o contador zera ao reiniciar o container) |
| Latencia p95 (s) | `histogram_quantile(0.95, sum(rate(vitorium_request_latency_seconds_bucket[1m])) by (le, endpoint))` | p95 por rota em segundos; comparar a linha de `/predict` com 0,05 s |
| Taxa de Erro do /predict (%) | `100 * (sum(rate(vitorium_requests_total{job="vitorium-api",endpoint="/predict",status_code=~"5.."}[5m])) or vector(0)) / sum(rate(vitorium_requests_total{job="vitorium-api",endpoint="/predict"}[5m]))` | Percentual de 5xx **só da inferência** em 5 min, numerador e denominador com o mesmo filtro (health checks e scrapes não diluem a taxa); o `or vector(0)` mostra 0% em vez de "No data" |
| Requisicoes por Segundo | `sum(rate(vitorium_requests_total[1m])) by (endpoint)` | Carga por rota; inclui os scrapes de `/metrics` e o healthcheck |
| Predicoes por Classe de Urgencia | `sum(vitorium_predictions_total) by (urgencia)` | Contagem acumulada por classe; base visual para o alerta de distribuição |

## 4. Alertas

As expressões exatas estão em `docker/alerts.yml`. A tabela resume a
semântica.

| Alerta | Condição | Limiar | Duração (`for`) | Severidade | Playbook |
|---|---|---|---|---|---|
| `VitoriumApiDown` | Prometheus não consegue coletar `/metrics` (`up == 0`) | `up == 0` | 1 min | critical | [link](#vitoriumapidown) |
| `VitoriumHighErrorRate` | Fração de respostas 5xx sobre o total em 5 min | > 5% | 5 min | critical | [link](#vitoriumhigherrorrate) |
| `VitoriumHighLatencyP95` | p95 de `/predict` pelo histograma | > 50 ms | 10 min | warning | [link](#vitoriumhighlatencyp95) |
| `VitoriumPredictionDistributionShift` | Fração de qualquer classe na última 1 h comparada ao prior de treino (0,345 / 0,323 / 0,333), com volume mínimo de predições | desvio absoluto > 0,15 | 30 min | warning | [link](#vitoriumpredictiondistributionshift) |

Severidade: `critical` exige ação imediata (serviço fora ou degradado para o
usuário); `warning` exige investigação no mesmo dia útil.

## 5. Playbooks

Todos os comandos assumem a raiz do repositório e a stack local. No Windows,
use `curl.exe` em vez de `curl` no PowerShell.

### VitoriumApiDown

**Sintomas.** Alerta *firing* em <http://localhost:9090/alerts>; target
`vitorium-api` como DOWN em <http://localhost:9090/targets>; painéis do
Grafana param de atualizar; `/predict` não responde.

**Diagnóstico.**

```bash
docker compose -f docker/docker-compose.yml ps            # api está "Up (healthy)"? reiniciando?
docker compose -f docker/docker-compose.yml logs api --tail 100
curl -s http://localhost:8000/health                       # esperado: {"status":"ok","model_loaded":true,"backend":"onnx"}
curl -s http://localhost:8000/metrics | grep vitorium_ | head
```

PromQL: `up{job="vitorium-api"}` e `changes(up{job="vitorium-api"}[15m])`
(oscilação indica container em loop de reinício).

Causas comuns: container parado ou em *crash loop* (erro de import, falta de
memória); porta 8000 ocupada; API de pé mas Prometheus sem rota de rede até
`api:8000` (por exemplo, Prometheus subido fora do Compose).

**Mitigação.**

1. `docker compose -f docker/docker-compose.yml restart api` e acompanhar
   `ps` até `healthy`.
2. Se a falha começou após uma nova imagem, fazer rollback (seção 7).
3. Se o log mostrar `Model artifact not found`, a API sobe mas `/predict`
   responde 503: esse caso aparece como `VitoriumHighErrorRate`, não como
   `ApiDown`. Reconstruir a imagem garantindo `models/model.joblib`.

**Quando escalar.** Se não voltar em 15 minutos com restart e rollback, ou
se o orçamento de erro mensal (216 min) estiver próximo do fim. Neste
projeto, escalar significa acionar o responsável pelo repositório; em um
hospital, a equipe de plataforma e o responsável clínico pelo fluxo de
triagem, que precisa saber que o apoio automático está indisponível.

### VitoriumHighErrorRate

**Sintomas.** Mais de 5% de respostas 5xx por 5 minutos; painel "Taxa de
Erro do /predict (%)" acima de 5; clientes recebendo 500 ou 503.

**Diagnóstico.**

```bash
docker compose -f docker/docker-compose.yml logs api --tail 200 | grep -iE "error|traceback|503"
curl -s http://localhost:8000/health                       # 503 + model_loaded false => 503 em todo /predict
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"texto": "Patient presents with acute chest pain radiating to the left arm, elevated troponin levels, and ECG changes consistent with myocardial infarction."}'
```

PromQL, para ver qual código e qual rota:

```promql
sum by (endpoint, status_code) (rate(vitorium_requests_total{status_code=~"5.."}[5m]))
```

500 de exceção não tratada aparecem no contador (seção 1.1); o stack trace
para o diagnóstico está no log do container.

**Mitigação.**

- `503` com `model_loaded: false`: artefato ausente ou corrompido na imagem.
  Rollback para a imagem anterior (seção 7) e corrigir o build.
- `500` após troca de backend de inferência: voltar para o backend sklearn
  com `VITORIUM_BACKEND=sklearn` (seção 7).
- `500` ligado a um tipo de entrada específico: reproduzir com o payload,
  corrigir validação no schema e publicar nova versão.

**Quando escalar.** Imediatamente, se a taxa passar de 50% ou se o
rollback não resolver em 15 minutos.

### VitoriumHighLatencyP95

**Sintomas.** p95 de `/predict` acima de 50 ms por 10 minutos; painel
"Latencia p95 (s)" acima de 0,05 para `/predict`.

**Diagnóstico.**

```promql
histogram_quantile(0.95, sum by (le) (rate(vitorium_request_latency_seconds_bucket{endpoint="/predict"}[5m])))
sum(rate(vitorium_requests_total{endpoint="/predict"}[5m]))
```

A segunda consulta mostra se a latência subiu junto com a carga (saturação)
ou sozinha (regressão).

```bash
docker stats --no-stream vitorium-api                      # CPU e memória do container
curl -s -o /dev/null -D - -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "Patient presents with acute chest pain radiating to the left arm, elevated troponin levels, and ECG changes consistent with myocardial infarction."}' | grep -i x-process-time-ms
```

O campo `latencia_ms` da resposta mede só a inferência; o header
`X-Process-Time-Ms` mede o processamento completo. Se `latencia_ms` está
baixo e o header alto, o gargalo não é o modelo.

**Mitigação.**

1. Carga alta: escalar horizontalmente (mais réplicas atrás de um balanceador
   ou mais workers do uvicorn). A API é *stateless*, então isso é seguro.
2. Regressão após deploy: rollback (seção 7).
3. Inferência lenta: confirmar que o backend ONNX está ativo; a inferência
   pura ONNX é 4,09x mais rápida que sklearn (média;
   `models/onnx_latency_comparison.json`) e o `/health` informa o backend
   ativo no campo `backend`.
4. Textos muito longos (até 20.000 caracteres permitidos): avaliar reduzir
   `max_length` em `src/api/schemas.py`.

**Quando escalar.** Se ultrapassar 250 ms p95 (borda de bucket) ou se vier
acompanhado de `VitoriumHighErrorRate`.

### VitoriumPredictionDistributionShift

**Sintomas.** A fração de alguma classe na última hora difere em mais de
0,15 do prior de treino por 30 minutos; painel "Predicoes por Classe de
Urgencia" claramente desequilibrado.

**Diagnóstico.**

```promql
sum by (urgencia) (increase(vitorium_predictions_total[1h]))
  / scalar(sum(increase(vitorium_predictions_total[1h])))
sum(increase(vitorium_predictions_total[1h]))
```

A primeira mostra a fração atual por classe; a segunda, o volume (com pouco
volume, a fração é ruidosa).

Perguntas a responder, nesta ordem:

1. **O tráfego é sintético ou repetitivo?** Testes de carga com um único
   texto produzem 100% de uma classe. É a causa mais provável em ambiente de
   demonstração.
2. **Excesso de `normal`?** É a assinatura de entrada fora da distribuição:
   texto em português, vazio ou sem vocabulário conhecido cai em `normal`
   por padrão (`docs/MODEL_CARD.md`, seção 7.3). Verificar com o cliente o
   idioma e o formato do que está sendo enviado.
3. **Mudança legítima de população?** Por exemplo, a integração passou a
   enviar só laudos de cardiologia. Nesse caso o prior mudou, não o modelo.

**Mitigação.**

- Tráfego de teste: nenhuma ação no modelo; registrar e silenciar a janela.
- Entrada fora da distribuição: corrigir a integração (idioma, campo
  errado) ou bloquear a entrada; retreinar não resolve.
- Mudança persistente e legítima: coletar amostra rotulada, medir F1 nela e,
  se houver queda, acionar retreino com dados novos (seção 6).

**Quando escalar.** Se persistir por mais de 24 horas sem explicação de
tráfego, ou se a amostra rotulada mostrar queda de recall de `urgente`.

## 6. Estratégia de deriva e política de retreino

**Honestidade sobre deriva.** Sem rótulos em produção, o sistema detecta
apenas sintomas (distribuição de predições), não perda de qualidade. A
confirmação exige a revisão de amostra rotulada da seção 1.2.

**Política de retreino.**

| Gatilho | Ação | Estado |
|---|---|---|
| Agenda semanal | DAG `vitorium_train_pipeline` (`schedule="@weekly"`, `dags/train_pipeline_dag.py`): `ingest_data` → `train_model` → `export_onnx` → `save_model_report` | Implementado e executado (`docs/evidence/airflow_dags_test.log`) |
| `VitoriumPredictionDistributionShift` persistente (mais de 24 h) com causa de dados confirmada | Disparo manual da DAG (`airflow dags trigger vitorium_train_pipeline`) depois de incorporar novos dados rotulados | Procedimento manual |
| Queda de F1 em amostra rotulada | Mesmo procedimento acima | Procedimento manual; a revisão de amostra é proposta |

Ressalva importante: hoje a DAG treina sempre sobre o mesmo corpus público
fixo. Retreinar sem dados novos reproduz o mesmo modelo e **não corrige
deriva**. O retreino só tem efeito quando novos exemplos rotulados entram em
`data/raw/`. Também não há portão de qualidade automático na DAG
(`docs/MODEL_CARD.md`, seção 9).

## 7. Rollback

Cada push na `main` publica duas tags da imagem no GHCR
(`.github/workflows/ci.yml`): `:latest` e `:<sha do commit>`. O rollback é
voltar para o `sha` anterior conhecido como bom.

```bash
# 1. Escolher o commit bom anterior
git log --oneline -5

# 2. Baixar a imagem daquele commit
docker pull ghcr.io/vdfs89/vitorium-mednlp-mlops:<sha>

# 3a. Rodar diretamente
docker run -d --name vitorium-api-rollback -p 8000:8000 ghcr.io/vdfs89/vitorium-mednlp-mlops:<sha>

# 3b. Ou reaproveitar o Compose: retaguear como a imagem que o serviço usa e subir sem build
docker tag ghcr.io/vdfs89/vitorium-mednlp-mlops:<sha> vitorium-api:latest
docker compose -f docker/docker-compose.yml up -d --no-build api
```

(Em 3a, pare antes o container `vitorium-api` para liberar a porta 8000.)

**Chave de fallback de backend.** Com o backend ONNX, a variável de ambiente
`VITORIUM_BACKEND=sklearn` faz a API voltar a servir com o pipeline
scikit-learn original, sem trocar de imagem:

```bash
docker run -d -p 8000:8000 -e VITORIUM_BACKEND=sklearn ghcr.io/vdfs89/vitorium-mednlp-mlops:latest
```

No Compose, o equivalente é definir `VITORIUM_BACKEND=sklearn` em
`environment` do serviço `api` e rodar
`docker compose -f docker/docker-compose.yml up -d api`. Use essa chave
primeiro quando o problema surgir logo após a troca para ONNX; use o rollback
de imagem para qualquer outra regressão.

Após qualquer rollback: confirmar `/health`, observar 15 minutos de painéis
e registrar o incidente (o que aconteceu, impacto, causa, ação preventiva).
