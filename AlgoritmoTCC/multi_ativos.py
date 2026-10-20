import argparse
import contextlib
import dataclasses
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp

UNIVERSO = {
    "PETR4.SA": "Petróleo e gás", "VALE3.SA": "Mineração", "ITUB4.SA": "Bancos",
    "BBDC4.SA": "Bancos", "BBAS3.SA": "Bancos", "ABEV3.SA": "Bebidas",
    "WEGE3.SA": "Bens de capital", "B3SA3.SA": "Serviços financeiros",
    "SUZB3.SA": "Papel e celulose", "GGBR4.SA": "Siderurgia", "MGLU3.SA": "Varejo",
    "ELET3.SA": "Energia elétrica", "VIVT3.SA": "Telecomunicações",
    "PRIO3.SA": "Petróleo e gás", "LREN3.SA": "Varejo",
}
UNIVERSO_PADRAO = list(UNIVERSO)[:10]


# ------------------------------------------------------------------------------
# Correções para múltiplos testes
# ------------------------------------------------------------------------------
def holm(p):
    """Holm-Bonferroni: controla a probabilidade de qualquer falso positivo."""
    p = np.asarray(p, float)
    m, ordem, adj, corrente = len(p), np.argsort(p), np.empty(len(p)), 0.0
    for i, idx in enumerate(ordem):
        corrente = max(corrente, (m - i) * p[idx])
        adj[idx] = min(1.0, corrente)
    return adj


def benjamini_hochberg(p):
    """BH: controla a proporção esperada de falsos positivos entre os achados."""
    p = np.asarray(p, float)
    m, adj, corrente = len(p), np.empty(len(p)), 1.0
    for rank, idx in zip(range(m, 0, -1), np.argsort(p)[::-1]):
        corrente = min(corrente, p[idx] * m / rank)
        adj[idx] = corrente
    return adj


# ------------------------------------------------------------------------------
# Análise de um ativo
# ------------------------------------------------------------------------------
def analisar_ativo(lm, cfg_base, ticker, progresso=None):
    cfg = dataclasses.replace(cfg_base, ticker=ticker, salvar_modelo_final=False)
    vol = cfg.tarefa == "volatilidade"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df = lm.montar_features(cfg)
        res = lm.executar_walk_forward(df, cfg, progresso=progresso)
        tabela, resumo, ex = (lm.relatorio_vol if vol else lm.relatorio_retorno)(res, cfg)
    y = res["y"]
    base = {"ativo": ticker, "setor": UNIVERSO.get(ticker, ""), "n_teste": len(y)}
    if vol:
        lstm = tabela.iloc[0]
        har = tabela[tabela["modelo"] == "har"].iloc[0]
        b = ex["boot"][ex["boot"]["vs"] == "har"].iloc[0]
        linha = {**base, "RMSE_LSTM": lstm["RMSE(log-vol)"], "RMSE_HAR": har["RMSE(log-vol)"],
                 "ganho_vs_HAR_%": lstm["ganho_vs_HAR_%"], "dif_perda": b["dif_perda_media"],
                 "IC95_inf": b["IC95_inf"], "IC95_sup": b["IC95_sup"], "p_boot": b["p_boot"]}
        # teste unilateral: "a LSTM tem perda MENOR que o HAR"
        linha["p_unilateral"] = b["p_boot"] / 2 if b["dif_perda_media"] < 0 else 1 - b["p_boot"] / 2
        return linha, None
    lstm, bh = tabela.iloc[0], tabela.iloc[1]
    b = ex["boot"].iloc[0]
    taxa = 100 * np.mean(y > 0)
    linha = {**base, "taxa_alta_%": taxa, "acc_LSTM_%": lstm["acc_%"],
             "acc_baseline_%": max(taxa, 100 - taxa), "dif_acc_pp": lstm["acc_%"] - max(taxa, 100 - taxa),
             "p_vs_maioria": lstm["p_vs_maioria"], "IC": lstm["IC"],
             "sharpe_LSTM": lstm["sharpe"], "sharpe_B&H": bh["sharpe"],
             "dif_sharpe": b["dif_sharpe"], "IC95_inf": b["IC95_inf"], "IC95_sup": b["IC95_sup"],
             "p_boot": b["p_boot"]}
    serie = pd.DataFrame({"LSTM": ex["rets"]["LSTM (ensemble)"], "BH": ex["rets"]["sempre_comprado"]},
                         index=res["datas"])
    return linha, serie


# ------------------------------------------------------------------------------
# Agregação entre ativos
# ------------------------------------------------------------------------------
def agregar(tabela, series, cfg):
    vol = cfg.tarefa == "volatilidade"
    m = len(tabela)
    ag = {"n_ativos": m}
    if vol:
        p = tabela["p_unilateral"].values
        melhor = tabela["ganho_vs_HAR_%"].values > 0
        ag["metrica"] = "ganho_vs_HAR_%"
        ag["media"] = float(tabela["ganho_vs_HAR_%"].mean())
    else:
        p = tabela["p_vs_maioria"].values
        melhor = tabela["dif_acc_pp"].values > 0
        ag["metrica"] = "dif_acc_pp"
        ag["media"] = float(tabela["dif_acc_pp"].mean())
    tabela["p_holm"] = holm(p)
    tabela["p_BH"] = benjamini_hochberg(p)
    ag["n_melhor"] = int(melhor.sum())
    ag["p_sinais"] = binomtest(int(melhor.sum()), m, 0.5, alternative="greater").pvalue
    col = "ganho_vs_HAR_%" if vol else "dif_acc_pp"
    ag["p_ttest"] = float(ttest_1samp(tabela[col], 0.0).pvalue) if m >= 3 else np.nan
    ag["n_signif_holm"] = int((tabela["p_holm"] < 0.05).sum())
    ag["n_signif_BH"] = int((tabela["p_BH"] < 0.05).sum())

    carteira = None
    if not vol and series:
        from lstm_mercado_v2 import _sharpe, bootstrap_dif_sharpe
        ra = pd.concat([s["LSTM"] for s in series.values()], axis=1).mean(axis=1)
        rb = pd.concat([s["BH"] for s in series.values()], axis=1).mean(axis=1)
        obs, lo, hi, pb = bootstrap_dif_sharpe(ra.values, rb.values, cfg)
        ag.update({"sharpe_carteira_LSTM": float(_sharpe(ra.values)),
                   "sharpe_carteira_BH": float(_sharpe(rb.values)),
                   "dif_sharpe_carteira": obs, "IC95_inf_carteira": lo,
                   "IC95_sup_carteira": hi, "p_boot_carteira": pb})
        carteira = pd.DataFrame({"LSTM": np.exp(ra.cumsum()), "comprar_e_segurar": np.exp(rb.cumsum())})
    return ag, carteira


def executar_multi(lm, cfg_base, tickers, progresso=None):
    """progresso(fração 0-1, texto). Ativos sem dados são ignorados e listados."""
    linhas, series, ignorados = [], {}, {}
    n = len(tickers)
    for i, tk in enumerate(tickers):
        def prog(k, nf, msg, i=i, tk=tk):
            if progresso:
                progresso((i + k / nf) / n, f"{tk} ({i + 1}/{n}): {msg}")
        try:
            linha, serie = analisar_ativo(lm, cfg_base, tk, prog)
        except Exception as e:  # noqa: BLE001
            ignorados[tk] = f"{type(e).__name__}: {str(e)[:110]}"
            continue
        linhas.append(linha)
        if serie is not None:
            series[tk] = serie
    if progresso:
        progresso(1.0, "Concluído")
    if not linhas:
        return {"tabela": pd.DataFrame(), "series": {}, "ignorados": ignorados, "agregado": None, "carteira": None}
    tabela = pd.DataFrame(linhas)
    ag, carteira = agregar(tabela, series, cfg_base)
    return {"tabela": tabela, "series": series, "ignorados": ignorados, "agregado": ag, "carteira": carteira}


def imprimir(r, cfg):
    t, ag = r["tabela"], r["agregado"]
    vol = cfg.tarefa == "volatilidade"
    print("\n=== RESULTADOS POR ATIVO ===")
    print(t.round(4).to_string(index=False))
    if r["ignorados"]:
        print("\nAtivos ignorados (sem dados):")
        for k, v in r["ignorados"].items():
            print(f"  {k}: {v}")
    print("\n=== CONSISTÊNCIA ENTRE ATIVOS ===")
    alvo = "ganho sobre o HAR" if vol else "acurácia acima da classe mais frequente"
    print(f"LSTM melhor em {ag['n_melhor']} de {ag['n_ativos']} ativos ({alvo}); "
          f"teste de sinais p = {ag['p_sinais']:.3f}")
    print(f"Média de {ag['metrica']}: {ag['media']:.3f} | teste t entre ativos p = {ag['p_ttest']:.3f} "
          "(ativos são correlacionados: tome com cautela)")
    print(f"Ativos significativos a 5% após correção: Holm = {ag['n_signif_holm']}, "
          f"Benjamini-Hochberg = {ag['n_signif_BH']}")
    if "sharpe_carteira_LSTM" in ag:
        print(f"\nCarteira de pesos iguais: Sharpe LSTM {ag['sharpe_carteira_LSTM']:.2f} contra "
              f"{ag['sharpe_carteira_BH']:.2f} de comprar e segurar | dif. {ag['dif_sharpe_carteira']:.2f} "
              f"(IC95% {ag['IC95_inf_carteira']:.2f} a {ag['IC95_sup_carteira']:.2f}, p_boot {ag['p_boot_carteira']:.3f})")


def main():
    import lstm_mercado_v2 as lm
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*", help="ex.: PETR4.SA VALE3.SA (padrão: universo de 10 ativos)")
    ap.add_argument("--tarefa", default="classificacao", choices=["classificacao", "regressao", "volatilidade"])
    ap.add_argument("--horizonte", type=int, default=None)
    ap.add_argument("--rapido", action="store_true", help="1 modelo, 3 folds, 30 épocas (só para testar)")
    a = ap.parse_args()

    vol = a.tarefa == "volatilidade"
    h = a.horizonte or (5 if vol else 1)
    kw = dict(n_seeds=1, n_folds=3, epocas=30, n_boot=300) if a.rapido else {}
    cfg = lm.Config(tarefa=a.tarefa, horizonte=1 if vol else h, horizonte_vol=h if vol else 5, **kw)
    tickers = a.tickers or UNIVERSO_PADRAO
    print(f"Analisando {len(tickers)} ativos | tarefa={cfg.tarefa} | modo {'rápido' if a.rapido else 'completo'}")

    r = executar_multi(lm, cfg, tickers, progresso=lambda f, msg: print(f"[{f:4.0%}] {msg}"))
    if r["agregado"] is None:
        print("Nenhum ativo com dados.", r["ignorados"])
        return
    imprimir(r, cfg)
    Path(cfg.saida).mkdir(exist_ok=True)
    r["tabela"].to_csv(Path(cfg.saida) / f"multi_ativos_{cfg.tarefa}.csv", index=False)
    ag = r["agregado"]
    resumo = {"n_ativos": ag["n_ativos"], "ativos_melhores": ag["n_melhor"], "media_" + ag["metrica"]: ag["media"],
              "p_sinais": ag["p_sinais"], "signif_holm": ag["n_signif_holm"]}
    lm.registrar_execucao(dataclasses.replace(cfg, ticker=f"MULTI({ag['n_ativos']})"), resumo)


if __name__ == "__main__":
    main()
