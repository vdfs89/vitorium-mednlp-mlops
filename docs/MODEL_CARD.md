# Model Card — VITORIUM (classificador de urgência de textos médicos)

Formato inspirado em Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019).
Regra deste documento: todo número cita o arquivo de onde veio. Quando um
valor é derivado (calculado a partir de outro número), a conta está escrita
ao lado.

> **Aviso principal.** O VITORIUM é um projeto acadêmico de MLOps (FIAP
> Pós-Tech, Fase 3). Os rótulos de urgência são um **mapeamento heurístico**
> de categorias de doença, não uma triagem clínica (não é Manchester, não é
> ESI). O modelo **não deve ser usado para nenhuma decisão médica**.

## 1. Detalhes do modelo

| Item | Valor | Fonte |
|---|---|---|
| Nome | VITORIUM triage baseline | `src/training/train.py` |
| Tipo | Classificação de texto multiclasse (3 classes), um rótulo por texto | `src/training/train.py` |
| Arquitetura | `sklearn.pipeline.Pipeline`: `TfidfVectorizer` → `LogisticRegression` | `src/training/train.py` (`build_pipeline`) |
| TF-IDF | `max_features=20_000`, `ngram_range=(1, 2)`, `min_df=2`, `stop_words="english"` | `src/training/train.py` |
| Classificador | `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)` | `src/training/train.py` |
| Seed | 42 | `models/metrics.json` (`random_seed`) |
| Tempo de treino | 3,79 s (3,785 s) em CPU | `models/metrics.json` (`train_seconds`) |
| Artefatos | `models/model.joblib` (sklearn) e `models/model.onnx` (exportado via `skl2onnx`, `zipmap=False`) | `models/`, `scripts/export_onnx.py` |
| Versão do artefato | Artefato versionado no commit `550099e` (2026-09-17), coberto pelas tags `v0.1.0`/`v0.1.1`. A API se anuncia como `version="0.1.0"` | `git log -- models/`, `src/api/main.py`, `docs/VERSIONING.md` |
| Biblioteca | scikit-learn 1.9.1 na imagem Docker | `requirements.txt` |
| Licença dos dados | CC BY-SA 3.0 (herdada do corpus) | `README.md` |
| Responsável | Vitor Diogo Fonseca da Silva (projeto individual) | `pyproject.toml` (`authors`) |

**Saída da API** (`src/api/schemas.py`): `urgencia` (classe com maior
probabilidade), `confianca` (essa probabilidade máxima), `probabilidades`
(as três) e `latencia_ms` (tempo de `predict_proba`, sem overhead HTTP).
`confianca` é o `predict_proba` cru da regressão logística. Ela **não foi
calibrada** (ver seção 7).

## 2. Uso pretendido

**Uso primário.** Demonstrar, de ponta a ponta, uma esteira MLOps para um
modelo leve de NLP: treino reprodutível, API REST, container, CI/CD,
retreino orquestrado, monitoramento e otimização de latência. O caso de
negócio ("triagem de laudos em um hospital") é a moldura didática pedida
pelo enunciado da Fase 3.

**Usuários pretendidos.** Avaliadores da FIAP, recrutadores e engenheiros
que queiram inspecionar a esteira. Não há usuário clínico pretendido.

**Fora de escopo (usos que este modelo NÃO suporta):**

- Qualquer decisão clínica, priorização real de pacientes, triagem de
  pronto-atendimento ou apoio a diagnóstico.
- Laudos ou prontuários reais, em especial em **português**: o corpus de
  treino é 100% em inglês (ver seção 7.3). O campo da API se chama `texto`,
  mas isso não significa suporte a português.
- Textos com múltiplas condições em que se espera mais de um rótulo. O
  modelo sempre devolve exatamente uma classe.
- Uso da `confianca` como probabilidade calibrada de risco.

## 3. Dados de treino e avaliação

**Corpus:** Medical Abstracts TC Corpus (sebischair, NLPIR 2022), 14.438
**abstracts** de artigos médicos em inglês, rotulados em 5 categorias de
condição do paciente (fonte: `README.md`). O split treino/teste é o oficial
do corpus: 11.550 treino e 2.888 teste (`models/metrics.json`, campos
`train_rows` e `test_rows`).

**Mapeamento heurístico para urgência** (`src/training/preprocess.py`,
`URGENCY_MAP`; contagens em `README.md`):

| Urgência | Classes originais | Amostras | % do total |
|---|---|---:|---:|
| `urgente` | Cardiovascular diseases + Nervous system diseases | 4.976 | 34,5% |
| `atencao` | Neoplasms + Digestive system diseases | 4.657 | 32,3% |
| `normal` | General pathological conditions | 4.805 | 33,3% |

O conjunto de teste tem 995 `urgente`, 932 `atencao` e 961 `normal`
(`models/metrics.json`, campo `support`), coerente com a tabela por classe
original do `README.md` (610 + 385 = 995; 633 + 299 = 932; 961).

**Pré-processamento.** Apenas `strip()` no texto e o mapeamento de rótulo
(`map_urgency`). Tokenização, minúsculas e remoção de stop words em inglês
ficam a cargo do `TfidfVectorizer`.

**Por que essa proxy.** O corpus não tem rótulo de urgência. Preferimos um
proxy explícito e documentado (ADR-003 em `docs/SPEC.md`) a inventar rótulos
clínicos que o dado não sustenta. A justificativa didática do agrupamento
(cardiovascular/neurológico como "padrão de emergência", neoplasias e
digestivo como "acompanhamento") é uma simplificação: há neoplasias
emergenciais e quadros cardiovasculares crônicos estáveis.

## 4. Métricas de avaliação

Avaliação em holdout único (split de teste oficial do corpus, 2.888
amostras), sem validação cruzada. Fonte de todos os números desta seção:
`models/metrics.json`.

**Métricas globais**

| Métrica | Valor |
|---|---:|
| Acurácia | 0,6195 |
| F1 macro | 0,6158 |
| Precisão macro | 0,6131 |
| Recall macro | 0,6200 |
| F1 ponderado | 0,6154 |

Referência para leitura: um classificador que sempre previsse a classe
majoritária do teste (`urgente`) teria acurácia 995 / 2.888 = 0,3445
(derivado de `support` em `models/metrics.json`). O modelo fica bem acima
desse piso, mas longe de um desempenho aceitável para qualquer uso real.

**Métricas por classe**

| Classe | Precisão | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| `atencao` | 0,6899 | 0,7425 | 0,7152 | 932 |
| `normal` | 0,4890 | 0,4391 | 0,4627 | 961 |
| `urgente` | 0,6605 | 0,6784 | 0,6693 | 995 |

**Por que `normal` é a classe mais fraca.** `normal` corresponde à categoria
original *General pathological conditions*, que é um "guarda-chuva": abstracts
sobre inflamação, infecção, mecanismos gerais de doença, que frequentemente
citam órgãos, tumores ou eventos cardiovasculares. O vocabulário dessa classe
se sobrepõe ao das outras duas, e um modelo de saco de palavras não tem como
separar "texto genérico que menciona o coração" de "texto sobre doença
cardiovascular". O erro vai nos dois sentidos: recall de 0,4391 (mais da
metade dos `normal` reais vai para outra classe) e precisão de 0,4890 (quase
metade do que o modelo chama de `normal` pertence a outra classe). Não há
matriz de confusão salva em `models/`, então não afirmamos para qual classe
os erros vão em maior proporção.

**Distribuição de predições esperada.** A partir de `models/metrics.json`
dá para reconstruir quantas vezes cada classe foi prevista no teste
(previstos = recall × suporte / precisão):

| Classe | Acertos (recall × suporte) | Previstos | Fração prevista | Fração real no teste |
|---|---:|---:|---:|---:|
| `atencao` | 692 | 1.003 | 0,347 | 0,323 |
| `normal` | 422 | 863 | 0,299 | 0,333 |
| `urgente` | 675 | 1.022 | 0,354 | 0,345 |

Os acertos somam 1.789 (1.789 / 2.888 = 0,6195, igual à acurácia), o que
confirma a conta. Essa tabela é a referência usada no alerta de mudança de
distribuição (`docs/MONITORING.md`): com dados parecidos com os de teste, o
desvio máximo em relação ao prior de treino é cerca de 0,034, bem abaixo do
limiar de 0,15.

**Análise desagregada.** O corpus não traz metadados de paciente (idade,
sexo, origem), então não foi possível avaliar desempenho por subgrupo
demográfico. Os únicos fatores avaliáveis são a classe e as características
do texto (seção 8).

## 5. Desempenho operacional (latência)

| Medição | Valor | Condições | Fonte |
|---|---:|---|---|
| Inferência pura sklearn (média) | 1,084 ms | 300 chamadas intercaladas, 1 texto por chamada, CPU, sem HTTP | `models/onnx_latency_comparison.json` |
| Inferência pura ONNX Runtime (média) | 0,265 ms | idem | `models/onnx_latency_comparison.json` |
| Speedup inferência pura (média) | 4,09x | paridade com sklearn em 2.888 textos: 100% dos rótulos, máx. dif. de probabilidade 1,6e-7 | `models/onnx_latency_comparison.json` |
| API end-to-end p50, sklearn vs ONNX | 3,401 ms vs 2,514 ms | 2 servidores uvicorn, 5 rodadas intercaladas × 300 req (warmup 50), mediana das rodadas | `models/api_latency_comparison.json` |
| API end-to-end p95, sklearn vs ONNX | 3,115 ms (ONNX) vs 4,486 ms (sklearn) | idem | `models/api_latency_comparison.json` |
| Speedup end-to-end p50 | 1,35x (−0,887 ms) | ONNX teve p50 menor nas 5 rodadas | `models/api_latency_comparison.json` |
| Baseline da Etapa 1, API p50 / p95 / p99 (sklearn) | 2,72 / 3,07 / 3,67 ms | 200 req, set/2026, outra sessão de máquina: não comparar diretamente com as linhas acima | `models/latency_baseline.json` |

Leitura honesta: o ganho de 4,09x vale para a chamada do modelo isolada. Na
API, a maior parte do tempo é FastAPI, validação Pydantic, serialização JSON
e HTTP; por isso o ganho end-to-end (1,35x no p50) é bem menor.

Errata: até a versão `v0.1.1` o repositório publicava 7,70x de speedup. Esse
número foi medido com uma exportação ONNX padrão do skl2onnx que **não era
equivalente** ao pipeline sklearn (descartava as stop words e mudava o
padrão de tokens; ~1,2% dos rótulos divergiam no teste) e que dependia do
locale `en_US.UTF-8`, ausente na imagem `python:3.11-slim`. O exportador foi
reescrito (`src/training/onnx_export.py`), a exportação agora falha se a
paridade quebrar, e os números foram medidos de novo. Todos
os benchmarks rodaram em uma única máquina de desenvolvimento, com um único
texto de exemplo (`scripts/bench_latency.py`, `SAMPLE_LAUDO`), e não
representam carga concorrente nem ambiente de nuvem.

## 6. Considerações éticas

- **Risco de sub-triagem silenciosa.** O erro mais grave em triagem é
  classificar como `normal` algo urgente. O modelo erra `normal` com
  frequência (precisão 0,4890) e, como mostrado na seção 8, devolve `normal`
  por padrão para textos que ele não entende. Em um contexto real isso seria
  inaceitável; aqui é a principal razão para o aviso de "não usar
  clinicamente".
- **Dados pessoais.** O corpus é composto de abstracts publicados, sem dados
  de pacientes. A API não persiste o texto recebido: o código de
  `src/api/main.py` não registra o conteúdo de `texto` em log. Mesmo assim,
  qualquer implantação com laudos reais exigiria avaliação de LGPD, controle
  de acesso (a API não tem autenticação, `docs/SPEC.md` seção 1) e política
  de retenção.
- **Automação e responsabilidade.** Mesmo um modelo bom deveria atuar só
  como apoio, com decisão final humana e registro de discordâncias.
- **Atribuição.** O dataset é CC BY-SA 3.0; derivações dos dados precisam
  manter a mesma licença e citar os autores (seção 10).

## 7. Limitações, vieses e riscos

### 7.1 Rótulos heurísticos

A variável-alvo é uma regra fixa sobre categorias de doença, não um
julgamento clínico de urgência. O modelo aprende a reconhecer **assunto**
(cardiovascular, neoplasia, etc.), não gravidade. Um abstract sobre
hipertensão leve controlada e outro sobre infarto agudo recebem o mesmo
rótulo-alvo (`urgente`).

### 7.2 Abstracts não são laudos

Abstracts científicos têm estrutura (objetivo, métodos, resultados), registro
formal e vocabulário de pesquisa. Laudos e evoluções clínicas são curtos,
telegráficos, cheios de abreviações e de negações ("sem sinais de...").
Mesmo em inglês, laudos reais estão fora da distribuição de treino.

### 7.3 Corpus somente em inglês

O vocabulário do TF-IDF foi aprendido em inglês, com stop words em inglês.
Texto em português é **fora da distribuição**: quase nenhum token coincide
com o vocabulário. Quando nenhum token coincide, o vetor TF-IDF é todo zero e
a predição passa a depender só dos interceptos da regressão logística, que
favorecem `normal` (intercepto 0,219 contra -0,115 de `atencao` e -0,104 de
`urgente` no artefato atual). Resultado: **todo texto totalmente fora do
vocabulário recebe `normal`, com a mesma confiança (cerca de 0,41)**,
independentemente do conteúdo. Verificação reproduzível:

```bash
python -c "import joblib; p = joblib.load('models/model.joblib'); \
t = 'Paciente com dor toracica aguda irradiando para o braco esquerdo'; \
print(p[0].transform([t]).nnz, p.predict([t]), p.predict_proba([t]).round(3), p[1].intercept_)"
```

### 7.4 Classe `normal` fraca

F1 de 0,4627 (`models/metrics.json`), explicado na seção 4. Somado ao item
7.3, `normal` é tanto a classe menos confiável quanto o "destino padrão" de
entradas que o modelo não reconhece.

### 7.5 Sem estudo de calibração

Não há curva de calibração, Brier score nem ajuste (Platt/isotônico). O
`class_weight="balanced"` também desloca as probabilidades. `confianca` deve
ser lida como um escore de ordenação, não como probabilidade.

### 7.6 Deriva de vocabulário do TF-IDF

O vocabulário (até 20.000 termos) fica congelado no treino. Termos novos
(nomes de fármacos, siglas, doenças emergentes) são ignorados na inferência.
Não existe hoje métrica de taxa de tokens fora do vocabulário em produção;
ela está proposta em `docs/MONITORING.md`.

### 7.7 Avaliação limitada

Um único holdout, sem validação cruzada nem intervalo de confiança. Com
2.888 amostras de teste, diferenças pequenas (centésimos de F1) entre
versões do modelo não devem ser tratadas como significativas sem um teste
estatístico.

### 7.8 Saco de palavras

Bigramas capturam pouco contexto. Negação ("no evidence of myocardial
infarction"), temporalidade ("history of stroke") e intensidade não são
modeladas.

## 8. Cenários de falha concretos

| Cenário | Comportamento esperado | Mitigação |
|---|---|---|
| Texto com menos de 10 caracteres | HTTP 422 (validação `min_length=10` em `src/api/schemas.py`) | Já mitigado pela validação. 4xx não conta como erro no SLO de 5xx |
| Texto com mais de 20.000 caracteres | HTTP 422 (`max_length=20_000`) | Cliente deve truncar ou resumir antes de enviar |
| Texto curto mas válido ("chest pain", 10 caracteres) | Aceito; predição baseada em 1 ou 2 termos, instável e pouco informativa | Exigir comprimento mínimo maior ou devolver um aviso de baixa evidência quando houver poucos termos no vocabulário |
| Só espaços (10 ou mais) | Aceito pelo schema; vetor TF-IDF zerado; sempre `normal` (seção 7.3) | Validar `texto.strip()` no schema; rejeitar entrada sem conteúdo |
| Texto em português ou outro idioma | HTTP 200 com predição quase sempre `normal`, sem relação com o conteúdo (seção 7.3) | Detectar idioma e recusar (ou sinalizar) entradas não inglesas; retreinar com corpus em português antes de qualquer uso local |
| Texto com várias condições (ex.: paciente oncológico com AVC) | Uma única classe; a probabilidade se divide entre classes e a `confianca` cai | Expor as três probabilidades (a API já expõe) e tratar `confianca` baixa como "revisão humana"; modelo multirrótulo seria a solução estrutural |
| Erros de digitação ("myocardail") | Termo com erro vira token fora do vocabulário e é ignorado | Normalização ortográfica ou n-gramas de caracteres em uma versão futura |
| Abreviações clínicas (MI, CHF, CVA, AVC, IAM) | Só contam se aparecerem com `min_df >= 2` no corpus de treino; siglas em português nunca contam | Dicionário de expansão de siglas no pré-processamento |
| Negação ("no evidence of infarction") | Termos de infarto pesam para `urgente` mesmo negados | Detecção de negação (ex.: regras NegEx) ou modelo contextual |
| Modelo ausente no container | `/predict` responde 503 e `/health` mostra `model_loaded: false` (`src/api/main.py`) | Coberto pelo alerta de taxa de erro; ver playbook em `docs/MONITORING.md` |

## 9. Manutenção

- **Retreino.** DAG Airflow `vitorium_train_pipeline`, `schedule="@weekly"`,
  tarefas `ingest_data` → `train_model` → `save_model_report`
  (`dags/train_pipeline_dag.py`). Importante: a DAG retreina sobre o mesmo
  corpus público fixo. Sem dados novos rotulados, o retreino semanal
  funciona como verificação de reprodutibilidade, não como adaptação a
  deriva. A DAG não tem portão de qualidade (não compara o novo F1 com o
  anterior antes de substituir o artefato); recomendamos adicionar um, por
  exemplo bloquear a promoção se o F1 macro cair abaixo do valor atual
  (0,6158, `models/metrics.json`) menos uma tolerância definida.
- **Promoção para produção.** A API carrega `models/model.joblib` na
  inicialização. Um modelo novo só chega ao serviço por uma nova imagem: o
  CI (`.github/workflows/ci.yml`) roda lint → test → build e, em push na
  `main`, publica `ghcr.io/vdfs89/vitorium-mednlp-mlops:latest` e
  `:<sha do commit>`. Observação: no workflow atual o job `test` retreina o
  modelo e o job `build` copia esse `model.joblib` recém-treinado para a
  imagem, enquanto `models/model.onnx` e `models/metrics.json` vêm do
  repositório. Com seed fixa e as mesmas versões de biblioteca o resultado
  deve coincidir, mas isso não é verificado automaticamente; um teste que
  compare a saída sklearn e ONNX dentro do CI fecharia essa lacuna.
- **Monitoramento.** Métricas Prometheus, dashboard Grafana, regras de
  alerta e playbooks em `docs/MONITORING.md`.
- **Versionamento.** Tags SemVer anotadas (`docs/VERSIONING.md`); a imagem
  publicada por commit (`:<sha>`) é o ponto de rollback.
- **Atualização deste documento.** Qualquer novo treino que altere
  `models/metrics.json` exige revisar as seções 4 e 7.

## 10. Como citar o dataset

Schopf, T., Braun, D., Matthes, F. *Evaluating Unsupervised Text
Classification: Zero-shot and Similarity-based Approaches.* NLPIR 2022
(International Conference on Natural Language Processing and Information
Retrieval). Dados:
<https://github.com/sebischair/Medical-Abstracts-TC-Corpus>, licença
CC BY-SA 3.0.

```bibtex
@inproceedings{schopf2022evaluating,
  title     = {Evaluating Unsupervised Text Classification: Zero-shot and Similarity-based Approaches},
  author    = {Schopf, Tim and Braun, Daniel and Matthes, Florian},
  booktitle = {Proceedings of the International Conference on Natural Language Processing and Information Retrieval (NLPIR 2022)},
  year      = {2022}
}
```
