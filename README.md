# TCC

# Previsão de mercado com LSTM: avaliação rigorosa em ações da B3

Pipeline em Python que testa se uma rede neural **LSTM** prevê a **direção** e a **volatilidade** de ações da B3 melhor do que métodos simples, e se a diferença é distinguível do acaso. O projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC).

> **Escopo desta versão:** o repositório faz **avaliação histórica fora da amostra** (walk-forward). Ele **não gera previsões para dias futuros** nem sinais de operação.

## Perguntas de pesquisa

- **H1.** A LSTM supera o baseline na previsão de direção (acurácia acima da classe mais frequente)?
- **H2.** A LSTM supera o modelo HAR na previsão de volatilidade?
- **H0 (para ambas).** Não há diferença distinguível do acaso.

## O que o código faz

1. Baixa dados diários do **Yahoo Finance** (preços ajustados por dividendos e desdobramentos) e, opcionalmente, Brent, dólar e Ibovespa.
2. Valida a qualidade dos dados: datas duplicadas, buracos acima de 7 dias, saltos diários acima de 40% (em log) e barra parcial do dia.
3. Cria 15 variáveis estacionárias (+3 exógenas): retornos, razões de médias móveis, volatilidades em várias escalas, RSI, MACD, Bollinger, ATR e volume relativo.
4. Treina a LSTM em **walk-forward** (janela expansiva), com embargo, early stopping e ensemble de modelos com sementes diferentes.
5. Compara com baselines: sempre comprado, momentum, reversão, Ridge e gradient boosting (direção); média histórica, persistência, EWMA, HAR, Ridge e gradient boosting (volatilidade).
6. Calcula métricas, simula uma estratégia com custos e faz **bootstrap em blocos** para medir a incerteza.
7. Em vários ativos, aplica **correção para múltiplos testes** (Holm e Benjamini-Hochberg) e monta uma carteira de pesos iguais.

## Estrutura do repositório

```
.
├── lstm_mercado_v2.py     # motor: dados, features, modelo, walk-forward, métricas, backtest
├── multi_ativos.py        # mesma análise em vários ativos + consistência entre ativos
├── app_streamlit.py       # painel web opcional (Streamlit)
├── iniciar_painel.bat     # atalho para abrir o painel no Windows
├── requirements.txt
└── saida/                 # gerada na execução (ignorada pelo Git)
```

## Instalação

Requer **Python 3.12** (o TensorFlow ainda não funciona bem com versões mais novas).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\python.exe -m pip install -r requirements.txt
# Linux/macOS:
.venv/bin/python -m pip install -r requirements.txt
```

## Como usar

**Um ativo** (padrão: PETR4.SA). Edite a classe `Config` no início do arquivo e rode:

```bash
python lstm_mercado_v2.py
```

**Vários ativos** (universo padrão de 10 papéis de setores diferentes):

```bash
python multi_ativos.py                                  # direção, modo completo
python multi_ativos.py --tarefa volatilidade --horizonte 5
python multi_ativos.py PETR4.SA VALE3.SA ITUB4.SA       # ativos específicos
python multi_ativos.py --rapido                         # apenas para testar o fluxo
```

**Painel web:**

```bash
python -m streamlit run app_streamlit.py
```

### Principais parâmetros (`Config`)

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `ticker` | `PETR4.SA` | Código do Yahoo Finance |
| `inicio` / `fim` | `2015-01-01` / hoje | Período. **Fixe `fim` para reproduzir os resultados** |
| `tarefa` | `classificacao` | `classificacao`, `regressao` ou `volatilidade` |
| `horizonte` / `horizonte_vol` | 1 / 5 | Dias à frente |
| `lookback` | 30 | Janela de observação |
| `treino_inicial`, `n_folds` | 0,5 / 5 | Walk-forward |
| `frac_validacao` | 0,15 | Final do treino usado no early stopping |
| `n_seeds` | 3 | Modelos por fold (ensemble) |
| `unidades_lstm`, `dropout` | 32 / 0,3 | Arquitetura |
| `custo_bps`, `long_short` | 10 / `False` | Backtest |
| `bloco_bootstrap`, `n_boot` | 20 / 2000 | Inferência |

## Arquivos gerados em `saida/`

| Arquivo | Conteúdo |
|---|---|
| `resultados_<tarefa>.csv` | Tabela de métricas por modelo |
| `multi_ativos_<tarefa>.csv` | Resultados por ativo, com p-valores corrigidos |
| `coleta_<ativo>.json` | Fonte, data/hora da coleta e período baixado |
| `historico_execucoes.csv` | Registro de cada configuração testada |
| `backtest.png`, `volatilidade.png`, `loss.png` | Gráficos |
| `lstm_final.keras`, `scaler.joblib`, `meta.json` | Modelo treinado com todo o histórico, só para arquivo |

## Metodologia resumida

- **Alvos.** Direção: sinal do retorno logarítmico do próximo dia. Volatilidade: log da volatilidade realizada nos próximos 5 dias.
- **Modelo.** LSTM (32 unidades) → Dropout → Dense (16, ReLU, L2) → sigmoide ou linear. Adam, perda entropia cruzada (direção) ou Huber (volatilidade).
- **Sem vazamento de dados.** Padronização ajustada só no treino, validação cronológica dentro de cada fold e embargo de h − 1 dias para horizontes maiores que 1.
- **Métricas.** Direção: acurácia, p-valor binomial (contra 50% e contra a classe mais frequente) e IC de Spearman. Volatilidade: RMSE e MAE em log-volatilidade, R² fora da amostra, ganho sobre o HAR e QLIKE. Estratégia: retorno, Sharpe anualizado, drawdown, exposição e trocas, com sensibilidade a custos.
- **Inferência.** Bootstrap em blocos circulares (IC de 95%) e correção para múltiplos testes entre ativos.ores que 1.
Métricas. Direção: acurácia, p-valor binomial (contra 50% e contra a classe mais frequente) e IC de Spearman. Volatilidade: RMSE e MAE em log-volatilidade, R² fora da amostra, ganho sobre o HAR e QLIKE. Estratégia: retorno, Sharpe anualizado, drawdown, exposição e trocas, com sensibilidade a custos.
Inferência. Bootstrap em blocos circulares (IC de 95%) e correção para múltiplos testes entre ativos.
