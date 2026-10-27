# Regra de versionamento — VITORIUM

Convenção de tags Git para este repositório, seguindo o mesmo padrão de
SemVer usado no TwinRankAI (`v2.0.1`).

## Formato

```
vMAJOR.MINOR.PATCH
```

Tag anotada sempre (`git tag -a vX.Y.Z -m "mensagem"`), nunca leve
(`git tag vX.Y.Z` sem `-a`) — a mensagem da tag é o changelog daquele ponto.

## Quando incrementar cada número

- **MAJOR** — só em marcos de entrega completa do desafio. `v1.0.0` é
  reservado para quando as 4 etapas estiverem prontas, testadas, e o vídeo
  STAR gravado (entrega final da Fase 3). Antes disso, tudo fica em `v0.x.y`.
- **MINOR** — ao concluir uma Etapa inteira (1, 2, 3 ou 4) com seus
  entregáveis obrigatórios funcionando de ponta a ponta, não só escritos.
  Ex.: `v0.2.0` = Etapa 2 completa (CI/CD rodando de verdade em um push real
  + DAG validada com `airflow dags test`).
- **PATCH** — correções, ajustes de dependência, documentação, ou pequenas
  melhorias dentro de uma etapa já fechada, sem adicionar escopo novo.

## Regra de ouro

**Nunca tagueie um estado quebrado.** Antes de qualquer tag:

```bash
make lint && make test
```

Se qualquer um falhar, a tag não acontece — corrige primeiro.

## Tags já não podem ser movidas

Uma tag publicada é imutável. Se algo errado foi tagueado, cria uma nova
tag de patch corrigindo (`v0.1.1`), nunca `git tag -f` sobre uma tag que já
foi empurrada (`git push --tags`) — isso quebra o histórico de quem já
puxou o repositório.

## Estado atual deste repositório

| Tag | O que cobre |
|---|---|
| `v0.1.0` | Scaffold inicial + SPEC.md (SDD) e esta regra de versionamento |
| `v0.1.1` | Migração para Poetry; Docker build e Compose (API + Prometheus + Grafana) verificados de ponta a ponta na máquina do Vitor |
| `v0.4.0` (a criar após o CI verde) | Etapas 2, 3 e 4 verificadas: DAG executada no Airflow, alertas testados com promtool, API servindo ONNX com paridade e comparação de latência end-to-end, Dockerfile multi-stage, Model Card e plano de monitoramento. Pula `v0.2.0`/`v0.3.0` porque as três etapas fecharam no mesmo dia — a regra de MINOR por etapa é sobre histórico legível, e uma tag por etapa apontando para o mesmo commit não acrescentaria informação |
| `v1.0.0` (reservada) | Entrega final: vídeo STAR gravado |

Comando, só depois do CI verde no commit final:

```bash
git tag -a v0.4.0 -m "Etapas 2-4 verificadas: Airflow, alertas, ONNX end-to-end"
git push origin v0.4.0
```
