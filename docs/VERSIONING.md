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
| `v0.1.0` | Scaffold inicial: Etapa 1 completa e testada (API + Docker + decisão de arquitetura); Etapas 2-4 com fundação escrita e parcialmente validada (ver `docs/SPEC.md` seção 2 para o status exato de cada requisito) |

Próximas tags esperadas, seguindo a regra acima: `v0.2.0` quando o CI/CD e a
DAG do Airflow forem validados de ponta a ponta em um ambiente real,
`v0.3.0` quando o Docker Compose com Prometheus/Grafana subir e gerar o
dashboard com tráfego real, `v0.4.0` ao fechar a comparação de latência
end-to-end do ONNX, e `v1.0.0` na entrega final com o vídeo STAR.
