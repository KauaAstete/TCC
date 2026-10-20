"""
Interface Streamlit para o algoritmo LSTM de previsão de mercado.

Rodar (na pasta onde estão lstm_mercado_v2.py e este arquivo):
    streamlit run app_streamlit.py
"""
import contextlib
import dataclasses
import datetime as dt
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

os.chdir(Path(__file__).resolve().parent)

st.set_page_config(page_title="Painel LSTM – previsão de mercado", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Carregando o TensorFlow (na primeira vez demora cerca de 30 s)...")
def carregar_modulo():
    import lstm_mercado_v2 as modulo
    return modulo


lm = carregar_modulo()


@st.cache_resource(show_spinner=False)
def carregar_multi():
    import multi_ativos as modulo
    return modulo


ma = carregar_multi()

TAREFAS = {"Direção (sobe ou desce)": "classificacao", "Retorno": "regressao", "Volatilidade": "volatilidade"}
EXOGENAS = {"Brent": ("brent", "BZ=F"), "Dólar (USD/BRL)": ("usd", "BRL=X"), "Ibovespa": ("ibov", "^BVSP")}
HIST_PATH = Path("saida") / "historico_execucoes.csv"


# ==============================================================================
# BARRA LATERAL: CONFIGURAÇÃO
# ==============================================================================
with st.sidebar:
    st.title("📈 Painel LSTM")

    st.subheader("Dados")
    ticker = st.text_input("Ativo (código do Yahoo Finance)", "PETR4.SA").strip().upper() or "PETR4.SA"
    inicio = st.date_input("Usar dados a partir de", value=dt.date(2015, 1, 1),
                           min_value=dt.date(2000, 1, 1), max_value=dt.date.today())
    st.caption("Fonte única: Yahoo Finance, com preços ajustados por dividendos e desdobramentos.")
    exog_sel = st.multiselect(
        "Séries extras (opcional)", list(EXOGENAS),
        help="Baixadas do Yahoo Finance: Brent (BZ=F), dólar (BRL=X) e Ibovespa (^BVSP).")

    st.subheader("O que prever")
    tarefa = TAREFAS[st.selectbox("Alvo", list(TAREFAS))]
    vol = tarefa == "volatilidade"
    horizonte = st.selectbox("Horizonte (dias à frente)", [5, 10, 20] if vol else [1, 5, 10, 20])
    lookback = st.slider("Janela de observação (dias)", 10, 60, 30, 5)

    st.subheader("Validação e modelo")
    rapido = st.checkbox("Modo rápido (para testar)", help="1 modelo por fold, 3 folds e até 30 épocas.")
    n_folds = 3 if rapido else st.slider("Folds do walk-forward", 2, 8, 5)
    n_seeds = 1 if rapido else st.slider("Modelos no ensemble", 1, 7, 3)
    epocas = 30 if rapido else 100

    st.subheader("Backtest")
    custo = st.slider("Custo por operação (bps)", 0, 30, 10, disabled=vol)
    long_short = st.checkbox("Permitir posições vendidas", disabled=vol)

    with st.expander("Avançado"):
        limite_salto = st.slider("Sinalizar variações diárias acima de", 0.2, 0.8, 0.4, 0.05,
                                 help="Em log. 0,40 ≈ queda de 33%. Pode indicar split/bonificação sem ajuste.")
        corrigir = st.checkbox("Remover saltos do histórico", help="Reescala os preços anteriores a cada salto acima do limite.")
        n_boot = 300 if rapido else st.slider("Reamostragens do bootstrap", 300, 5000, 2000, 100)
        salvar_final = st.checkbox("Salvar modelo final em saida/", help="Treina um modelo com todo o histórico. Demora mais.")

    st.divider()
    bt_validar = st.button("Validar dados")
    bt_rodar = st.button("Executar análise", type="primary")
    bt_atualizar = st.button("Baixar dados de novo", help="Descarta o cache e busca tudo outra vez no Yahoo Finance.")


def montar_cfg():
    return lm.Config(
        ticker=ticker, inicio=str(inicio),
        exogenas={EXOGENAS[k][0]: EXOGENAS[k][1] for k in exog_sel},
        limite_salto=limite_salto, corrigir_saltos=corrigir,
        tarefa=tarefa, horizonte=1 if vol else horizonte, horizonte_vol=horizonte if vol else 5,
        lookback=lookback, n_folds=n_folds, n_seeds=n_seeds, epocas=epocas, n_boot=n_boot,
        custo_bps=float(custo), long_short=long_short and not vol, salvar_modelo_final=salvar_final,
        saida="saida")


# ==============================================================================
# EXECUÇÃO
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def carregar_dados(ticker, inicio, tarefa, horizonte, horizonte_vol, limite_salto, corrigir, exog):
    cfg = lm.Config(ticker=ticker, inicio=inicio, exogenas=dict(exog), limite_salto=limite_salto,
                    corrigir_saltos=corrigir, tarefa=tarefa, horizonte=horizonte, horizonte_vol=horizonte_vol)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df = lm.montar_features(cfg)
    return df, buf.getvalue()


def executar(cfg, df):
    barra = st.progress(0.0, text="Iniciando...")

    def progresso(k, n, msg):
        barra.progress(min(k / n, 1.0), text=msg)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = lm.executar_walk_forward(df, cfg, progresso=progresso)
        if cfg.tarefa == "volatilidade":
            tabela, resumo, extras = lm.relatorio_vol(res, cfg)
        else:
            tabela, resumo, extras = lm.relatorio_retorno(res, cfg)
        lm.registrar_execucao(cfg, resumo)
        if cfg.salvar_modelo_final:
            lm.treinar_e_salvar_final(df, cfg)
    barra.empty()
    return {"cfg": cfg, "res": res, "tabela": tabela, "resumo": resumo, "extras": extras, "log": buf.getvalue()}


if bt_atualizar:
    lm.limpar_cache()
    carregar_dados.clear()
    st.toast("Cache descartado. Os dados serão baixados de novo do Yahoo Finance.")

if bt_validar or bt_rodar:
    st.session_state.pop("erro", None)
    cfg = montar_cfg()
    try:
        with st.spinner("Carregando e validando os dados..."):
            df, log = carregar_dados(ticker, str(inicio), tarefa, cfg.horizonte, cfg.horizonte_vol,
                                     limite_salto, corrigir, tuple(cfg.exogenas.items()))
        st.session_state["dados"] = {"df": df, "log": log, "ticker": ticker}
        if bt_rodar:
            st.session_state["resultado"] = executar(cfg, df)
    except Exception as e:  # noqa: BLE001
        st.session_state["erro"] = f"{type(e).__name__}: {e}"


def n_execucoes():
    return len(pd.read_csv(HIST_PATH)) if HIST_PATH.exists() else 1


# ==============================================================================
# TELAS
# ==============================================================================
def tela_dados():
    d = st.session_state.get("dados")
    if not d:
        st.info("Escolha o ativo na barra lateral e clique em **Validar dados** para checar a qualidade do histórico.")
        return
    df = d["df"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Amostras", f"{len(df):,}".replace(",", "."))
    c2.metric("Features", df.shape[1] - 2)
    c3.metric("Período", f"{df.index[0]:%m/%Y} a {df.index[-1]:%m/%Y}")
    st.subheader("Trajetória do preço (índice, início = 100)")
    st.line_chart(pd.DataFrame({"preço": 100 * np.exp(df["ret_1"].cumsum())}))
    st.caption("Reconstruída a partir dos retornos. Saltos bruscos indicam split, bonificação ou erro nos dados.")
    st.subheader("Relatório de validação")
    st.code(d["log"], language=None)


def veredito_retorno(r):
    tabela, ex = r["tabela"], r["extras"]
    lstm, bh = tabela.iloc[0], tabela.iloc[1]
    ic = ex["boot"].iloc[0]
    limiar = 0.05 / max(n_execucoes(), 1)
    motivos = []
    if lstm["p_vs_maioria"] >= limiar:
        motivos.append(f"p-valor contra a classe mais frequente = {lstm['p_vs_maioria']:.3f} (limiar {limiar:.4f})")
    if lstm["sharpe"] <= bh["sharpe"]:
        motivos.append(f"Sharpe {lstm['sharpe']:.2f} contra {bh['sharpe']:.2f} de comprar e segurar")
    if ic["IC95_inf"] <= 0:
        motivos.append("o IC95% da diferença de Sharpe inclui 0")
    if motivos:
        st.warning("**Sem evidência de sinal.** " + "; ".join(motivos) + ".")
    else:
        st.success("**Há indício de sinal.** Passou nos três critérios. Confirme em outros ativos e períodos "
                   "antes de confiar, lembrando que cada configuração testada aumenta a chance de acaso.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acurácia da LSTM", f"{lstm['acc_%']:.1f}%", f"{lstm['acc_%'] - bh['acc_%']:+.1f} pp vs sempre comprado")
    c2.metric("Correlação (IC)", f"{lstm['IC']:.3f}")
    c3.metric("Sharpe da LSTM", f"{lstm['sharpe']:.2f}", f"{lstm['sharpe'] - bh['sharpe']:+.2f} vs comprar e segurar")
    c4.metric("Retorno no período", f"{lstm['retorno_%']:.0f}%", f"{lstm['retorno_%'] - bh['retorno_%']:+.0f} pp vs comprar e segurar")


def veredito_vol(r):
    tabela, ex = r["tabela"], r["extras"]
    lstm = tabela.iloc[0]
    har = tabela[tabela["modelo"] == "har"].iloc[0]
    b = ex["boot"]
    b = b[b["vs"] == "har"].iloc[0]
    if b["IC95_sup"] < 0:
        st.success("**A LSTM supera o HAR** de forma estatisticamente distinguível (IC95% da diferença de perda abaixo de 0).")
    elif b["IC95_inf"] > 0:
        st.warning("**A LSTM é pior que o HAR** de forma estatisticamente distinguível.")
    else:
        st.info("**Sem diferença distinguível entre a LSTM e o HAR.** O IC95% da diferença de perda inclui 0.")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE da LSTM (log-vol)", f"{lstm['RMSE(log-vol)']:.3f}", f"HAR: {har['RMSE(log-vol)']:.3f}", delta_color="off")
    c2.metric("Ganho sobre o HAR", f"{lstm['ganho_vs_HAR_%']:.1f}%")
    c3.metric("QLIKE da LSTM", f"{lstm['QLIKE']:.3f}", f"HAR: {har['QLIKE']:.3f}", delta_color="off")


def tela_resultados():
    r = st.session_state.get("resultado")
    if not r:
        st.info("Clique em **Executar análise** na barra lateral. O treino leva alguns minutos; use o modo rápido para testar.")
        return
    cfg, res, tabela, ex = r["cfg"], r["res"], r["tabela"], r["extras"]
    alvo = f"volatilidade em {cfg.horizonte_vol} dias" if cfg.tarefa == "volatilidade" else f"{cfg.tarefa}, {cfg.horizonte} dia(s)"
    st.caption(f"Última execução: {cfg.ticker} · {alvo} · {len(res['y'])} pregões de teste "
               f"({res['datas'][0]:%d/%m/%Y} a {res['datas'][-1]:%d/%m/%Y})")
    if cfg.tarefa == "volatilidade":
        veredito_vol(r)
        ens = np.mean(res["seeds"], axis=0)
        anual = lambda x: np.exp(x) * np.sqrt(252) * 100
        st.subheader("Volatilidade anualizada: realizada contra previsões")
        st.line_chart(pd.DataFrame({"realizada": anual(res["y"]), "HAR": anual(res["har"]),
                                    "LSTM": anual(ens)}, index=res["datas"]))
    else:
        veredito_retorno(r)
        st.subheader("Capital acumulado no teste (com custos)")
        eq = ex["equities"]
        st.line_chart(pd.DataFrame({n: eq[n] for n in ("LSTM (ensemble)", "sempre_comprado", "ridge", "grad_boosting")},
                                   index=res["datas"]))
    st.subheader("Comparação com os baselines")
    st.dataframe(tabela.round(4), hide_index=True)
    st.download_button("Baixar tabela (CSV)", tabela.to_csv(index=False), f"resultados_{cfg.tarefa}.csv", "text/csv")


def tela_multi():
    st.markdown("Roda a mesma análise em **vários ativos da B3** e testa se a LSTM supera o baseline de forma "
                "**consistente**, com correção para múltiplos testes. Usa as opções da barra lateral "
                "(alvo, horizonte, folds etc.). Com vários ativos, use o modo rápido para testar primeiro.")
    escolhidos = st.multiselect("Ativos", list(ma.UNIVERSO), default=ma.UNIVERSO_PADRAO,
                                format_func=lambda t: f"{t} ({ma.UNIVERSO[t]})")
    extra = st.text_input("Outros códigos (separados por vírgula)", placeholder="ex.: RENT3.SA, RADL3.SA")
    tickers = escolhidos + [t for t in dict.fromkeys(x.strip().upper() for x in extra.split(",")) if t and t not in escolhidos]
    st.caption("Cada ativo é baixado do Yahoo Finance. Muitos ativos de uma vez podem atingir o limite de requisições do Yahoo; se isso ocorrer, aguarde alguns minutos e execute de novo.")

    if tickers and st.button("Executar em todos os ativos", type="primary"):
        cfg = montar_cfg()
        barra = st.progress(0.0, text="Iniciando...")
        r = ma.executar_multi(lm, cfg, tickers, progresso=lambda f, msg: barra.progress(min(f, 1.0), text=msg))
        barra.empty()
        if r["agregado"] is not None:
            ag = r["agregado"]
            resumo = {"n_ativos": ag["n_ativos"], "ativos_melhores": ag["n_melhor"],
                      "media_" + ag["metrica"]: ag["media"], "p_sinais": ag["p_sinais"],
                      "signif_holm": ag["n_signif_holm"]}
            with contextlib.redirect_stdout(io.StringIO()):
                lm.registrar_execucao(dataclasses.replace(cfg, ticker=f"MULTI({ag['n_ativos']})"), resumo)
        st.session_state["multi"] = {"r": r, "cfg": cfg}

    m = st.session_state.get("multi")
    if not m:
        st.info("Escolha os ativos e clique em **Executar em todos os ativos**.")
        return
    r, cfg = m["r"], m["cfg"]
    vol = cfg.tarefa == "volatilidade"
    if r["ignorados"]:
        st.warning("Ativos ignorados (sem dados): " + "; ".join(f"{k} ({v[:70]})" for k, v in r["ignorados"].items()))
    if r["agregado"] is None:
        st.error("Não foi possível baixar nenhum ativo do Yahoo Finance. Confira os códigos (ex.: VALE3.SA) e a conexão.")
        return
    ag, t = r["agregado"], r["tabela"]
    mt = ag["n_ativos"]
    if ag["p_sinais"] < 0.05 and ag["n_signif_BH"] >= 1:
        st.success(f"**Há consistência entre ativos.** A LSTM foi melhor em {ag['n_melhor']} de {mt} e pelo menos um "
                   "ativo segue significativo após a correção. Confirme em outros períodos antes de concluir.")
    elif ag["p_sinais"] < 0.05 or ag["n_signif_BH"] >= 1:
        st.info(f"**Evidência parcial.** A LSTM foi melhor em {ag['n_melhor']} de {mt} ativos, mas nem todos os critérios "
                "de consistência foram atendidos.")
    else:
        st.warning(f"**Sem evidência consistente.** A LSTM foi melhor em {ag['n_melhor']} de {mt} ativos, o que é "
                   "compatível com o acaso, e nenhum ativo é significativo após a correção para múltiplos testes.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ativos com LSTM melhor", f"{ag['n_melhor']} de {mt}")
    c2.metric("Média de " + ("ganho sobre o HAR (%)" if vol else "dif. de acurácia (pp)"), f"{ag['media']:.2f}")
    c3.metric("Significativos (Holm / BH)", f"{ag['n_signif_holm']} / {ag['n_signif_BH']}")
    c4.metric("Teste de sinais (p)", f"{ag['p_sinais']:.3f}")
    col = "ganho_vs_HAR_%" if vol else "dif_acc_pp"
    st.subheader("Desempenho por ativo" + (" (ganho sobre o HAR, %)" if vol else " (acurácia da LSTM menos a do baseline, pp)"))
    st.bar_chart(t.set_index("ativo")[col])
    if r["carteira"] is not None:
        st.subheader("Carteira de pesos iguais com os sinais da LSTM")
        st.line_chart(r["carteira"])
        st.caption(f"Sharpe da carteira: LSTM {ag['sharpe_carteira_LSTM']:.2f} contra {ag['sharpe_carteira_BH']:.2f} de comprar e segurar. "
                   f"Diferença {ag['dif_sharpe_carteira']:.2f} (IC95% {ag['IC95_inf_carteira']:.2f} a {ag['IC95_sup_carteira']:.2f}).")
    st.subheader("Tabela completa")
    st.dataframe(t.round(4), hide_index=True)
    st.caption("p_holm e p_BH são os p-valores corrigidos para múltiplos testes entre os ativos. "
               "Os ativos são correlacionados entre si, então os testes agregados devem ser lidos com cautela.")
    st.download_button("Baixar tabela (CSV)", t.to_csv(index=False), f"multi_ativos_{cfg.tarefa}.csv", "text/csv")


def tela_previsao():
    st.markdown("Treina a LSTM com **todo o histórico** e estima o que deve ocorrer nos próximos pregões após a última "
                "data disponível. Usa as opções da barra lateral (ativo, alvo, horizonte, modelos no ensemble). "
                "Rode **depois do fechamento** (a B3 fecha às 18h): a barra parcial do dia é descartada.")
    ult = st.session_state.get("resultado")
    if ult and ult["cfg"].ticker == ticker and ult["cfg"].tarefa == tarefa:
        t = ult["tabela"]
        if tarefa == "volatilidade":
            st.info(f"Última validação deste ativo e alvo: a LSTM teve ganho de {t.iloc[0]['ganho_vs_HAR_%']:.1f}% sobre o HAR.")
        else:
            st.info(f"Última validação deste ativo e alvo: acurácia de {t.iloc[0]['acc_%']:.1f}% contra {t.iloc[1]['acc_%']:.1f}% "
                    f"de sempre comprado (p contra a classe mais frequente = {t.iloc[0]['p_vs_maioria']:.3f}). "
                    "Se a LSTM não superou o baseline, esta previsão não deve ser tratada como sinal.")
    else:
        st.warning("Você ainda não rodou a validação para este ativo e alvo. Sem ela, não dá para saber se o modelo "
                   "tem alguma capacidade preditiva. Rode **Executar análise** primeiro.")

    if st.button("Gerar previsão", type="primary"):
        cfg = montar_cfg()
        barra = st.progress(0.0, text="Baixando dados e treinando...")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = lm.prever_amanha(cfg, progresso=lambda k, n, msg: barra.progress(min(k / n, 1.0), text=msg))
                lm.registrar_previsao(cfg, r)
            st.session_state["previsao"] = r
        except Exception as e:  # noqa: BLE001
            st.error(f"{type(e).__name__}: {e}")
        barra.empty()

    r = st.session_state.get("previsao")
    if r:
        v = lm.interpretar_previsao(r)
        st.caption(f"{r['ticker']} · dados até {r['data_ref']} · próximos {r['horizonte']} pregão(ões) · "
                   f"{r['n_modelos']} modelo(s) treinado(s) com {r['n_treino']} amostras")
        if r["tarefa"] == "classificacao":
            c1, c2, c3 = st.columns(3)
            c1.metric("Probabilidade de alta", f"{v['prob_alta_%']:.1f}%",
                      f"{v['prob_alta_%'] - v['taxa_alta_historica_%']:+.1f} pp vs taxa histórica ({v['taxa_alta_historica_%']:.1f}%)",
                      delta_color="off")
            c2.metric("Convicção", f"{v['conviccao_%']:.0f}%")
            c3.metric("Dispersão entre modelos", f"{v['dispersao_entre_modelos_pp']:.1f} pp")
            st.caption("Perto de 50% significa sem convicção. O que importa é a diferença para a taxa histórica de alta, "
                       "e ela costuma ser pequena.")
        elif r["tarefa"] == "regressao":
            c1, c2 = st.columns(2)
            c1.metric("Retorno esperado", f"{v['retorno_esperado_%']:+.2f}%")
            c2.metric("Ridge (baseline)", f"{v['ridge_retorno_%']:+.2f}%")
            st.caption("Retornos diários são quase imprevisíveis: espere valores muito próximos de zero.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Volatilidade anualizada", f"{v['vol_anualizada_%']:.1f}%")
            c2.metric("Volatilidade diária", f"{v['vol_diaria_%']:.2f}%")
            c3.metric("HAR (baseline)", f"{v['HAR_vol_anualizada_%']:.1f}%")
            st.caption(f"EWMA: {v['EWMA_vol_anualizada_%']:.1f}% · persistência: {v['persistência_vol_anualizada_%']:.1f}%.")

    st.subheader("Acompanhamento das previsões feitas")
    st.caption("Cada previsão fica salva. Depois que o prazo passa, confira com o que aconteceu: é o teste mais honesto, "
               "porque o modelo não conhecia o resultado. Poucas previsões dizem quase nada; só dezenas ou centenas importam.")
    if st.button("Conferir com o que aconteceu"):
        with st.spinner("Baixando preços e comparando..."):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    st.session_state["acomp"] = lm.atualizar_previsoes(montar_cfg())
            except Exception as e:  # noqa: BLE001
                st.error(f"{type(e).__name__}: {e}")
    p = st.session_state.get("acomp")
    if p is None and (Path("saida") / "previsoes.csv").exists():
        p = pd.read_csv(Path("saida") / "previsoes.csv")
    if p is not None and len(p):
        if "acertou_direcao" in p:
            ok = p["acertou_direcao"].dropna()
            if len(ok):
                st.metric("Acertos de direção", f"{ok.mean():.0%}", f"{len(ok)} previsão(ões) já resolvida(s)", delta_color="off")
        st.dataframe(p.iloc[::-1], hide_index=True)
    else:
        st.info("Nenhuma previsão salva ainda.")


def tela_detalhes():
    r = st.session_state.get("resultado")
    if not r:
        st.info("Os detalhes estatísticos aparecem depois de uma execução.")
        return
    cfg, ex, res = r["cfg"], r["extras"], r["res"]
    if cfg.tarefa == "volatilidade":
        st.subheader("Bootstrap: diferença de perda (LSTM menos baseline)")
        st.dataframe(ex["boot"].round(5), hide_index=True)
        st.caption("Valores negativos favorecem a LSTM. Se o IC95% inclui 0, não há diferença distinguível.")
        st.subheader("Estabilidade por fold (RMSE em log-vol)")
    else:
        st.subheader("Bootstrap: Sharpe da estratégia menos Sharpe de comprar e segurar")
        st.dataframe(ex["boot"].round(3), hide_index=True)
        st.caption("Se o IC95% inclui 0, não há evidência de que a estratégia supere comprar e segurar.")
        st.subheader("Sharpe conforme o custo de transação")
        st.dataframe(ex["sens"].round(3), hide_index=True)
        st.subheader("Estabilidade por fold (regimes de mercado)")
    st.dataframe(ex["folds"].round(3), hide_index=True)
    st.subheader("Curva de loss (último modelo treinado)")
    h = res["hist"]
    st.line_chart(pd.DataFrame({"treino": h["loss"], "validação": h["val_loss"]}))
    with st.expander("Log completo da execução"):
        st.code(r["log"], language=None)


def tela_historico():
    if not HIST_PATH.exists():
        st.info("O histórico aparece depois da primeira execução.")
        return
    h = pd.read_csv(HIST_PATH)
    n = len(h)
    st.metric("Configurações testadas", n)
    st.caption(f"Cada configuração testada aumenta a chance de um resultado bom por puro acaso. "
               f"Com {n}, use um limiar de p-valor mais rígido (cerca de {0.05 / n:.4f}).")
    st.dataframe(h.iloc[::-1], hide_index=True)
    st.download_button("Baixar histórico (CSV)", h.to_csv(index=False), "historico_execucoes.csv", "text/csv")


def tela_ajuda():
    st.markdown("""
**Como ler os resultados**

- **Acurácia direcional**: porcentagem de vezes que o modelo acertou se o mercado subiria ou cairia. Compare com *sempre comprado*, não com 50%.
- **p vs. maioria**: probabilidade de acertar tanto por acaso, dado que o mercado tem mais dias de um lado. Só abaixo do limiar (0,05 dividido pelo número de configurações testadas) há evidência.
- **IC**: correlação entre a previsão e o retorno real. Perto de zero significa que não há relação.
- **Sharpe**: retorno por unidade de risco, depois dos custos. A LSTM precisa superar o de *comprar e segurar*.
- **Bootstrap**: reamostra blocos de dias para estimar a incerteza. Se o intervalo de 95% inclui zero, a diferença pode ser ruído.
- **Volatilidade**: os baselines são persistência (a vol. recente continua), EWMA (RiskMetrics) e HAR (regressão em vol. de 5, 20 e 60 dias). Nas métricas de erro, menor é melhor.

**Cuidados**

- Se o retorno de comprar e segurar parecer alto demais, abra a aba **Dados** e procure saltos no preço.
- Os preços vêm do Yahoo Finance já ajustados por dividendos e desdobramentos. O Yahoo pode revisar dados passados, então guarde a data da coleta (arquivo `saida/coleta_<ativo>.json`).
- Este painel é uma ferramenta de estudo, não recomendação de investimento.
""")


st.title("Previsão de mercado com LSTM")
if st.session_state.get("erro"):
    st.error(st.session_state["erro"])
    st.caption("Se o erro citar ImpersonateError, rode `python -m pip install -U yfinance curl_cffi` e tente de novo. Se persistir, aguarde alguns minutos (limite de requisições do Yahoo).")

aba1, aba2, aba3, aba4, aba5, aba6, aba7 = st.tabs(
    ["Resultados", "Previsão", "Multi-ativos", "Dados", "Detalhes estatísticos", "Histórico", "Ajuda"])
with aba1:
    tela_resultados()
with aba2:
    tela_previsao()
with aba3:
    tela_multi()
with aba4:
    tela_dados()
with aba5:
    tela_detalhes()
with aba6:
    tela_historico()
with aba7:
    tela_ajuda()
