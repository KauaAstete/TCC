import json
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from scipy.stats import binomtest, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
@dataclass
class Config:
    # Dados
    ticker: str = "PETR4.SA"
    inicio: str = "2015-01-01"
    fim: Optional[str] = None  # None = até hoje
    exogenas: dict = field(default_factory=lambda: {
        "brent": "BZ=F", "usd": "BRL=X", "ibov": "^BVSP"})
    limite_salto: float = 0.40     # variação diária (log) acima disso é sinalizada
    corrigir_saltos: bool = False  # True = remove saltos (splits/bonificações) do histórico

    # Problema: "classificacao" (sobe/desce) | "regressao" (retorno) | "volatilidade"
    tarefa: str = "classificacao"
    horizonte: int = 1             # dias à frente (classificacao/regressao)
    horizonte_vol: int = 5         # dias à frente para a volatilidade realizada
    lookback: int = 30

    # Walk-forward
    treino_inicial: float = 0.5
    n_folds: int = 5
    frac_validacao: float = 0.15
    n_seeds: int = 3
    seed: int = 42

    # Modelo
    unidades_lstm: int = 32
    unidades_dense: int = 16
    dropout: float = 0.3
    l2: float = 1e-4
    lr: float = 1e-3
    epocas: int = 100
    paciencia: int = 12
    batch_size: int = 32
    ridge_alpha: float = 10.0

    # Backtest e inferência
    custo_bps: float = 10.0
    custos_sensibilidade: tuple = (0.0, 10.0, 20.0)
    long_short: bool = False
    bloco_bootstrap: int = 20
    n_boot: int = 2000

    # Saída
    saida: str = "saida"
    salvar_modelo_final: bool = True


# ==============================================================================
# 1. DADOS, VALIDAÇÃO E FEATURES
# ==============================================================================
CACHE_TTL_S = 6 * 3600     # downloads reaproveitados por 6 h (evita limite de requisições do Yahoo)
_CACHE = {}


def limpar_cache():
    """Esquece os downloads em memória: a próxima chamada baixa tudo de novo do Yahoo Finance."""
    _CACHE.clear()


def _baixar_yahoo(ticker, inicio, fim):
    """ÚNICA fonte de dados: Yahoo Finance via yfinance, com preços ajustados por dividendos
    e desdobramentos (auto_adjust=True). Se a identidade de navegador padrão do yfinance falhar
    (ImpersonateError), tenta outras antes de desistir."""
    for alvo in (None, "chrome", "chrome124", "chrome120", "safari"):
        try:
            kw = {}
            if alvo:
                from curl_cffi import requests as cr
                kw["session"] = cr.Session(impersonate=alvo)
            df = yf.download(ticker, start=inicio, end=fim, auto_adjust=True,
                             progress=False, **kw)
            if isinstance(df.columns, pd.MultiIndex):      # yfinance recente devolve MultiIndex
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                idx = pd.DatetimeIndex(df.index)           # datas sem fuso/hora (alinha ativos e exógenas)
                df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
                return df.dropna(how="all")
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame()


def baixar(ticker, inicio, fim):
    chave = (ticker, inicio, fim)
    agora = time.time()
    if chave in _CACHE and agora - _CACHE[chave][0] < CACHE_TTL_S:
        return _CACHE[chave][1].copy()
    df = _baixar_yahoo(ticker, inicio, fim)
    if not df.empty:
        df.attrs["coletado_em"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        _CACHE[chave] = (agora, df)
    return df.copy()


def registrar_coleta(cfg, ts, series):
    """Grava saida/coleta_<ativo>.json: fonte, data/hora da coleta e período baixado.
    Guarde esse arquivo: o Yahoo pode revisar dados passados, então a data importa."""
    saida = Path(cfg.saida)
    saida.mkdir(exist_ok=True)
    nome = cfg.ticker.replace("^", "").replace("=", "_")
    (saida / f"coleta_{nome}.json").write_text(json.dumps({
        "fonte": f"Yahoo Finance (yfinance {yf.__version__})",
        "precos": "ajustados por dividendos e desdobramentos (auto_adjust=True)",
        "coletado_em": ts, "inicio_solicitado": cfg.inicio, "fim_solicitado": cfg.fim or "hoje",
        "series": series}, indent=2, ensure_ascii=False))


def validar_dados(df, cfg):
    """Diagnóstico de qualidade. Um retorno de buy-and-hold implausível quase sempre
    vem de saltos (split/bonificação) ou de linhas erradas."""
    print("\n--- VALIDAÇÃO DOS DADOS ---")
    dup = int(df.index.duplicated().sum())
    if dup:
        print(f"[aviso] {dup} datas duplicadas removidas")
        df = df[~df.index.duplicated(keep="last")]
    if (df["Close"] <= 0).any():
        print("[aviso] preços <= 0 removidos")
        df = df[df["Close"] > 0]

    dias = df.index.to_series().diff().dt.days
    buracos = dias[dias > 7]
    if len(buracos):
        print(f"[aviso] {len(buracos)} buraco(s) > 7 dias (maior: {int(buracos.max())} dias, "
              f"em {buracos.idxmax().date()})")

    r = np.log(df["Close"] / df["Close"].shift(1))
    top = r.abs().sort_values(ascending=False).head(5).index
    print("Maiores variações diárias:", ", ".join(f"{d.date()} ({r[d] * 100:+.1f}%)" for d in top))

    saltos = r[r.abs() > cfg.limite_salto]
    if len(saltos):
        print(f"[aviso] {len(saltos)} variação(ões) acima de {cfg.limite_salto:.0%} (log): "
              "pode ser split/bonificação sem ajuste ou erro nos dados.")
        if cfg.corrigir_saltos:
            for d, v in saltos.items():
                antes = df.index < d
                for c in ("Open", "High", "Low", "Close"):
                    if c in df.columns:
                        df.loc[antes, c] = df.loc[antes, c] * np.exp(v)
            print("[corrigido] saltos removidos do histórico (preços anteriores reescalados).")
        else:
            print("           Para removê-los automaticamente: Config(corrigir_saltos=True).")

    anos = (df.index[-1] - df.index[0]).days / 365.25
    ini, fimp = df["Close"].iloc[0], df["Close"].iloc[-1]
    print(f"Preço {ini:.2f} → {fimp:.2f} | multiplicador {fimp / ini:.2f}x em {anos:.1f} anos "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def _remover_barra_parcial(df):
    """Se o pregão de hoje ainda pode estar aberto, a última barra do Yahoo é parcial: descarta."""
    agora = pd.Timestamp.now(tz="America/Sao_Paulo")
    if df.index[-1].date() == agora.date() and agora.hour < 19:
        print("[aviso] a barra de hoje pode estar incompleta (pregão aberto): descartada.")
        return df.iloc[:-1]
    return df


def montar_features(cfg, incluir_futuro=False):
    """incluir_futuro=True mantém as últimas linhas, que têm features mas ainda não têm alvo
    (é o que permite prever o próximo pregão)."""
    df = baixar(cfg.ticker, cfg.inicio, cfg.fim)
    if df.empty:
        raise RuntimeError(
            f"Não foi possível baixar {cfg.ticker} do Yahoo Finance. Confira o código do ativo "
            "(ex.: PETR4.SA) e a conexão. Se o erro citar ImpersonateError, rode "
            "'python -m pip install -U yfinance curl_cffi'. Se persistir, aguarde alguns minutos "
            "(limite de requisições do Yahoo) e tente de novo.")
    ts = df.attrs.get("coletado_em", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
    df = _remover_barra_parcial(df)
    resumo_serie = lambda tk, d: {"ticker": tk, "primeira_data": str(d.index[0].date()),
                                  "ultima_data": str(d.index[-1].date()), "linhas": int(len(d))}
    coleta = [resumo_serie(cfg.ticker, df)]
    print(f"Fonte: Yahoo Finance (yfinance {yf.__version__}) | preços ajustados | coletado em {ts}")
    df = validar_dados(df, cfg)
    close, high, low = df["Close"], df["High"], df["Low"]
    ret1 = np.log(close / close.shift(1))

    f = pd.DataFrame(index=df.index)
    f["ret_1"] = ret1
    f["ret_5"] = np.log(close / close.shift(5))
    f["ret_20"] = np.log(close / close.shift(20))
    f["sma_ratio_10"] = close / close.rolling(10).mean() - 1
    f["sma_ratio_30"] = close / close.rolling(30).mean() - 1

    # Volatilidade em várias escalas (base do modelo HAR)
    f["lvol_5"] = np.log(ret1.rolling(5).std())
    f["lvol_20"] = np.log(ret1.rolling(20).std())
    f["lvol_60"] = np.log(ret1.rolling(60).std())
    f["lvol_ewma"] = np.log(np.sqrt((ret1 ** 2).ewm(alpha=0.06, adjust=False, min_periods=20).mean()))
    f["lvol_pass"] = np.log(np.sqrt((ret1 ** 2).rolling(cfg.horizonte_vol).mean()))

    delta = close.diff()
    ganho = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    perda = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi_14"] = (100 * ganho / (ganho + perda)).mask((ganho + perda) == 0, 50)

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    m20, s20 = close.rolling(20).mean(), close.rolling(20).std()
    f["bb_pos"] = (close - m20) / (2 * s20)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / close
    vol = df["Volume"].replace(0, np.nan)
    f["vol_rel"] = np.log(vol / vol.rolling(20).mean()).fillna(0)

    for nome, tk in cfg.exogenas.items():
        try:
            bruto = baixar(tk, cfg.inicio, cfg.fim)
            s = bruto["Close"].reindex(df.index).ffill()
            f[f"{nome}_ret"] = np.log(s / s.shift(1))
            coleta.append(resumo_serie(tk, bruto))
        except Exception as e:  # noqa: BLE001
            print(f"[aviso] exógena '{nome}' ({tk}) indisponível: {e}")

    registrar_coleta(cfg, ts, coleta)

    # Alvos (as features em t só usam informação até t)
    if cfg.tarefa == "volatilidade":
        h = cfg.horizonte_vol
        rv = np.sqrt((ret1 ** 2).rolling(h).mean().shift(-h))   # vol. realizada em t+1..t+h
        f["alvo"] = np.log(rv.clip(lower=1e-6))
    else:
        h = cfg.horizonte
        f["alvo"] = ret1.rolling(h).sum().shift(-h)              # retorno acumulado t+1..t+h
    f["ret_seg"] = ret1.shift(-1)                                # retorno do dia seguinte (backtest)
    f = f.replace([np.inf, -np.inf], np.nan)
    if incluir_futuro:
        return f.dropna(subset=[c for c in f.columns if c not in ("alvo", "ret_seg")])
    return f.dropna()


# ==============================================================================
# 2. JANELAS, SPLITS E MODELO
# ==============================================================================
def janelas(X, idx_alvos, lookback):
    return np.stack([X[t - lookback + 1: t + 1] for t in idx_alvos]).astype("float32")


def gerar_folds(n, treino_inicial, n_folds):
    inicio = int(n * treino_inicial)
    tam = (n - inicio) // n_folds
    for k in range(n_folds):
        a = inicio + k * tam
        b = n if k == n_folds - 1 else a + tam
        yield a, b


def dividir_treino(fim_treino, lookback, frac_val):
    n_tr = fim_treino - (lookback - 1)
    n_val = max(int(n_tr * frac_val), 1)
    corte = fim_treino - n_val
    return corte, np.arange(lookback - 1, corte), np.arange(corte, fim_treino)


def alvo_lstm(y, cfg, mu, sd):
    if cfg.tarefa == "classificacao":
        return (y > 0).astype("float32")
    return ((y - mu) / sd).astype("float32")


def construir_modelo(lookback, n_feats, cfg):
    cls = cfg.tarefa == "classificacao"
    modelo = Sequential([
        Input(shape=(lookback, n_feats)),
        LSTM(cfg.unidades_lstm, dropout=cfg.dropout),
        Dropout(cfg.dropout),
        Dense(cfg.unidades_dense, activation="relu", kernel_regularizer=l2(cfg.l2)),
        Dense(1, activation="sigmoid" if cls else "linear"),
    ])
    loss = "binary_crossentropy" if cls else tf.keras.losses.Huber(delta=1.0)
    modelo.compile(optimizer=Adam(learning_rate=cfg.lr, clipnorm=1.0), loss=loss)
    return modelo


def treinar_lstm(X_fit, y_fit, X_val, y_val, cfg, seed):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    modelo = construir_modelo(X_fit.shape[1], X_fit.shape[2], cfg)
    cbs = [
        EarlyStopping(monitor="val_loss", patience=cfg.paciencia, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=max(cfg.paciencia // 2, 1), min_lr=1e-5),
    ]
    h = modelo.fit(X_fit, y_fit, validation_data=(X_val, y_val), epochs=cfg.epocas,
                   batch_size=cfg.batch_size, callbacks=cbs, verbose=0)
    return modelo, h


# ==============================================================================
# 3. WALK-FORWARD COM BASELINES (COM EMBARGO PARA HORIZONTE > 1)
# ==============================================================================
def executar_walk_forward(df, cfg, progresso=None):
    feats = [c for c in df.columns if c not in ("alvo", "ret_seg")]
    X_raw, y, r_seg = df[feats].values, df["alvo"].values, df["ret_seg"].values
    n, L = len(df), cfg.lookback
    vol = cfg.tarefa == "volatilidade"
    cls = cfg.tarefa == "classificacao"
    h = cfg.horizonte_vol if vol else cfg.horizonte
    ix = {c: feats.index(c) for c in ("ret_1", "lvol_5", "lvol_20", "lvol_60",
                                       "lvol_ewma", "lvol_pass")}
    chaves = ("idx", "y", "r_seg", "fold", "bench", "ret1", "ridge", "hgb", "pers", "ewma", "har")
    acum = {k: [] for k in chaves}
    seeds = [[] for _ in range(cfg.n_seeds)]
    hist = None

    for k, (a, b) in enumerate(gerar_folds(n, cfg.treino_inicial, cfg.n_folds)):
        if progresso:
            progresso(k, cfg.n_folds, f"Treinando fold {k + 1}/{cfg.n_folds}...")
        # Embargo: o rótulo do último dia de treino só pode usar dias <= a (início do teste)
        fim_tr = a - h + 1
        corte, t_fit, t_val = dividir_treino(fim_tr, L, cfg.frac_validacao)
        t_te, t_base = np.arange(a, b), np.arange(L - 1, fim_tr)

        scaler = StandardScaler().fit(X_raw[:corte])
        Xs = np.clip(scaler.transform(X_raw), -5, 5)

        ridge = Ridge(alpha=cfg.ridge_alpha).fit(Xs[t_base], y[t_base])
        acum["ridge"].append(ridge.predict(Xs[t_te]))
        if vol:
            hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                max_iter=200, random_state=0).fit(Xs[t_base], y[t_base])
            acum["hgb"].append(hgb.predict(Xs[t_te]))
            cols_har = [ix["lvol_5"], ix["lvol_20"], ix["lvol_60"]]
            har = LinearRegression().fit(Xs[t_base][:, cols_har], y[t_base])
            acum["har"].append(har.predict(Xs[t_te][:, cols_har]))
            acum["pers"].append(X_raw[t_te, ix["lvol_pass"]])
            acum["ewma"].append(X_raw[t_te, ix["lvol_ewma"]])
        else:
            hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                 max_iter=200, random_state=0).fit(Xs[t_base], y[t_base] > 0)
            acum["hgb"].append(hgb.predict_proba(Xs[t_te])[:, 1] - 0.5)
            acum["ret1"].append(X_raw[t_te, ix["ret_1"]])
        acum["idx"].append(t_te)
        acum["y"].append(y[t_te])
        acum["r_seg"].append(r_seg[t_te])
        acum["fold"].append(np.full(len(t_te), k))
        acum["bench"].append(np.full(len(t_te), y[:fim_tr].mean()))

        mu = y[t_fit].mean() if vol else 0.0
        sd = y[t_fit].std()
        X_fit, X_val, X_te = (janelas(Xs, t, L) for t in (t_fit, t_val, t_te))
        y_fit, y_val = alvo_lstm(y[t_fit], cfg, mu, sd), alvo_lstm(y[t_val], cfg, mu, sd)
        for s in range(cfg.n_seeds):
            modelo, hh = treinar_lstm(X_fit, y_fit, X_val, y_val, cfg, seed=cfg.seed + s)
            p = modelo.predict(X_te, verbose=0).ravel()
            seeds[s].append(p - 0.5 if cls else p * sd + mu)
            hist = hh.history

        ens = np.mean([sc[-1] for sc in seeds], axis=0)
        if vol:
            metrica = f"RMSE(log-vol) ensemble={np.sqrt(np.mean((y[t_te] - ens) ** 2)):.3f}"
        else:
            metrica = f"acc ensemble={np.mean((ens > 0) == (y[t_te] > 0)) * 100:.1f}%"
        print(f"fold {k + 1}/{cfg.n_folds}: treino até {df.index[fim_tr - 1].date()} | "
              f"teste {df.index[a].date()} → {df.index[b - 1].date()} | "
              f"épocas={len(hist['loss'])} | {metrica}")
        if progresso:
            progresso(k + 1, cfg.n_folds, f"Fold {k + 1}/{cfg.n_folds} concluído")

    res = {k: np.concatenate(v) for k, v in acum.items() if v}
    res["seeds"] = [np.concatenate(s) for s in seeds]
    res["hist"] = hist
    res["datas"] = df.index[res["idx"]]
    return res


# ==============================================================================
# 4. INFERÊNCIA: BOOTSTRAP EM BLOCOS
# ==============================================================================
def _indices_bloco(n, bloco, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / bloco))
    ini = rng.integers(0, n, size=(n_boot, nb))
    return ((ini[:, :, None] + np.arange(bloco)) % n).reshape(n_boot, -1)[:, :n]


def _sharpe(r, eixo=None):
    sd = r.std(axis=eixo)
    m = r.mean(axis=eixo)
    return np.where(sd > 0, m / np.where(sd > 0, sd, 1) * np.sqrt(252), np.nan)


def bootstrap_dif_sharpe(ra, rb, cfg):
    """IC95% e p-valor aproximado da diferença de Sharpe (a - b), reamostrando blocos."""
    idx = _indices_bloco(len(ra), cfg.bloco_bootstrap, cfg.n_boot)
    d = _sharpe(ra[idx], 1) - _sharpe(rb[idx], 1)
    d = d[~np.isnan(d)]
    obs = float(_sharpe(ra) - _sharpe(rb))
    return obs, np.percentile(d, 2.5), np.percentile(d, 97.5), 2 * min(np.mean(d <= 0), np.mean(d >= 0))


def bootstrap_media(d, bloco, n_boot):
    """IC95% e p-valor aproximado da média de d (ex.: diferença de perdas)."""
    idx = _indices_bloco(len(d), bloco, n_boot)
    m = d[idx].mean(axis=1)
    return float(d.mean()), np.percentile(m, 2.5), np.percentile(m, 97.5), 2 * min(np.mean(m <= 0), np.mean(m >= 0))


# ==============================================================================
# 5. MÉTRICAS E BACKTEST (DIREÇÃO / RETORNO)
# ==============================================================================
def backtest(score, r_seg, cfg, custo_bps=None):
    custo = cfg.custo_bps if custo_bps is None else custo_bps
    pos = np.where(score > 0, 1.0, -1.0 if cfg.long_short else 0.0)
    troca = np.abs(np.diff(pos, prepend=0.0))
    ret = pos * r_seg - troca * custo / 1e4
    equity = np.exp(np.cumsum(ret))
    pico = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return {
        "retorno_%": (equity[-1] - 1) * 100,
        "sharpe": float(_sharpe(ret)),
        "max_dd_%": (equity / pico - 1).min() * 100,
        "exposicao_%": np.mean(pos != 0) * 100,
        "trocas": troca.sum(),
    }, equity, ret


def avaliar(nome, score, y, r_seg, cfg, pred_ret=None, bench=None):
    n = len(y)
    acertos = int(np.sum((score > 0) == (y > 0)))
    taxa_alta = np.mean(y > 0)
    maioria = max(taxa_alta, 1 - taxa_alta)
    linha = {
        "modelo": nome,
        "acc_%": 100 * acertos / n,
        "p_vs_50%": binomtest(acertos, n, 0.5, alternative="greater").pvalue,
        "p_vs_maioria": binomtest(acertos, n, maioria, alternative="greater").pvalue,
        "IC": spearmanr(score, y)[0] if np.std(score) > 0 else np.nan,
    }
    if pred_ret is not None:
        linha["RMSE"] = np.sqrt(np.mean((y - pred_ret) ** 2))
        linha["MAE"] = np.mean(np.abs(y - pred_ret))
        linha["R2_oos"] = 1 - np.sum((y - pred_ret) ** 2) / np.sum((y - bench) ** 2)
    bt, equity, ret = backtest(score, r_seg, cfg)
    linha.update(bt)
    return linha, equity, ret


def relatorio_retorno(res, cfg):
    y, r, cls = res["y"], res["r_seg"], cfg.tarefa == "classificacao"
    ens = np.mean(res["seeds"], axis=0)
    modelos = {
        "LSTM (ensemble)": (ens, None if cls else ens),
        "sempre_comprado": (np.ones_like(y), None),
        "momentum": (res["ret1"], None),
        "reversao": (-res["ret1"], None),
        "ridge": (res["ridge"], res["ridge"]),
        "grad_boosting": (res["hgb"], None),
        "media_historica": (res["bench"], res["bench"]),
    }
    linhas, equities, rets, scores = [], {}, {}, {}
    for nome, (score, pred) in modelos.items():
        linha, eq, rd = avaliar(nome, score, y, r, cfg, pred, res["bench"])
        linhas.append(linha)
        equities[nome], rets[nome], scores[nome] = eq, rd, score
    tabela = pd.DataFrame(linhas)

    accs = [100 * np.mean((s > 0) == (y > 0)) for s in res["seeds"]]
    print("\n=== RESULTADOS FORA DA AMOSTRA (walk-forward, todos os folds) ===")
    print(f"Período: {res['datas'][0].date()} → {res['datas'][-1].date()} | {len(y)} pregões | "
          f"dias de alta no alvo: {100 * np.mean(y > 0):.1f}% | tarefa={cfg.tarefa}, horizonte={cfg.horizonte}d")
    print(f"Acurácia por seed: {np.round(accs, 2)} (média {np.mean(accs):.2f}% ± {np.std(accs):.2f})\n")
    print(tabela.round(4).to_string(index=False))
    if cfg.horizonte > 1:
        print("\n[nota] Com horizonte > 1 os alvos se sobrepõem: os p-valores binomiais ficam "
              "otimistas. Dê mais peso ao bootstrap abaixo.")

    # Sensibilidade a custos
    print("\n--- SHARPE X CUSTO DE TRANSAÇÃO (bps por unidade negociada) ---")
    sens = []
    for nome in ("LSTM (ensemble)", "ridge", "momentum", "sempre_comprado"):
        linha = {"modelo": nome}
        for c in cfg.custos_sensibilidade:
            linha[f"sharpe@{c:.0f}bps"] = backtest(scores[nome], r, cfg, c)[0]["sharpe"]
        sens.append(linha)
    df_sens = pd.DataFrame(sens)
    print(df_sens.round(3).to_string(index=False))

    # Bootstrap: cada estratégia contra comprar e segurar
    print(f"\n--- BOOTSTRAP EM BLOCOS ({cfg.n_boot} reamostragens, bloco={cfg.bloco_bootstrap}d): "
          "Sharpe da estratégia MENOS Sharpe do sempre_comprado ---")
    bs = []
    for nome in ("LSTM (ensemble)", "ridge", "grad_boosting", "momentum"):
        obs, lo, hi, p = bootstrap_dif_sharpe(rets[nome], rets["sempre_comprado"], cfg)
        bs.append({"modelo": nome, "dif_sharpe": obs, "IC95_inf": lo, "IC95_sup": hi, "p_boot": p})
    df_boot = pd.DataFrame(bs)
    print(df_boot.round(3).to_string(index=False))
    print("Leitura: se o IC95% inclui 0, não há evidência de que a estratégia supere comprar e segurar.")

    # Estabilidade por fold
    print("\n--- ESTABILIDADE POR FOLD (regimes de mercado) ---")
    por_fold = []
    for k in np.unique(res["fold"]):
        m = res["fold"] == k
        por_fold.append({
            "fold": int(k) + 1,
            "inicio": res["datas"][m][0].date(), "fim": res["datas"][m][-1].date(),
            "taxa_alta_%": 100 * np.mean(y[m] > 0),
            "acc_LSTM_%": 100 * np.mean((ens[m] > 0) == (y[m] > 0)),
            "acc_ridge_%": 100 * np.mean((res["ridge"][m] > 0) == (y[m] > 0)),
            "sharpe_LSTM": float(_sharpe(rets["LSTM (ensemble)"][m])),
            "sharpe_B&H": float(_sharpe(rets["sempre_comprado"][m])),
        })
    df_folds = pd.DataFrame(por_fold)
    print(df_folds.round(2).to_string(index=False))

    lstm = tabela.iloc[0]
    resumo = {"acc_%": lstm["acc_%"], "p_vs_maioria": lstm["p_vs_maioria"],
              "sharpe": lstm["sharpe"], "retorno_%": lstm["retorno_%"],
              "sharpe_bh": tabela.iloc[1]["sharpe"], "r2_oos": lstm.get("R2_oos", np.nan)}
    extras = {"equities": equities, "rets": rets, "sens": df_sens, "boot": df_boot, "folds": df_folds}
    return tabela, resumo, extras


# ==============================================================================
# 6. RELATÓRIO DE VOLATILIDADE
# ==============================================================================
def relatorio_vol(res, cfg):
    y = res["y"]
    ens = np.mean(res["seeds"], axis=0)
    modelos = {
        "LSTM (ensemble)": ens, "media_historica": res["bench"], "persistencia": res["pers"],
        "ewma": res["ewma"], "har": res["har"], "ridge": res["ridge"], "grad_boosting": res["hgb"],
    }
    mse_bench = np.mean((y - res["bench"]) ** 2)
    mse_har = np.mean((y - res["har"]) ** 2)
    linhas = []
    for nome, p in modelos.items():
        mse = np.mean((y - p) ** 2)
        razao = np.exp(2 * (y - p))                       # var_real / var_prevista
        linhas.append({
            "modelo": nome, "RMSE(log-vol)": np.sqrt(mse), "MAE": np.mean(np.abs(y - p)),
            "R2_vs_media": 1 - mse / mse_bench, "ganho_vs_HAR_%": 100 * (1 - mse / mse_har),
            "QLIKE": np.mean(razao - np.log(razao) - 1),
        })
    tabela = pd.DataFrame(linhas)

    print("\n=== PREVISÃO DE VOLATILIDADE FORA DA AMOSTRA (walk-forward) ===")
    print(f"Período: {res['datas'][0].date()} → {res['datas'][-1].date()} | {len(y)} pregões | "
          f"alvo: log da vol. realizada nos próximos {cfg.horizonte_vol} dias")
    print(f"{'Menor é melhor em RMSE/MAE/QLIKE; ganho_vs_HAR > 0 = melhor que o HAR.'}\n")
    print(tabela.round(4).to_string(index=False))

    print(f"\n--- BOOTSTRAP EM BLOCOS: diferença de perda quadrática (LSTM − baseline) ---")
    bloco = max(cfg.bloco_bootstrap, 2 * cfg.horizonte_vol)   # alvos se sobrepõem
    bs = []
    for nome in ("har", "ewma", "persistencia", "grad_boosting"):
        d = (y - ens) ** 2 - (y - modelos[nome]) ** 2
        obs, lo, hi, p = bootstrap_media(d, bloco, cfg.n_boot)
        bs.append({"vs": nome, "dif_perda_media": obs, "IC95_inf": lo, "IC95_sup": hi, "p_boot": p})
    df_boot = pd.DataFrame(bs)
    print(df_boot.round(5).to_string(index=False))
    print("Leitura: dif_perda_media < 0 com IC95% inteiro abaixo de 0 = LSTM significativamente melhor.")

    print("\n--- ESTABILIDADE POR FOLD (RMSE em log-vol) ---")
    por_fold = []
    for k in np.unique(res["fold"]):
        m = res["fold"] == k
        por_fold.append({
            "fold": int(k) + 1, "inicio": res["datas"][m][0].date(), "fim": res["datas"][m][-1].date(),
            "LSTM": np.sqrt(np.mean((y[m] - ens[m]) ** 2)),
            "HAR": np.sqrt(np.mean((y[m] - res["har"][m]) ** 2)),
            "persistencia": np.sqrt(np.mean((y[m] - res["pers"][m]) ** 2)),
        })
    df_folds = pd.DataFrame(por_fold)
    print(df_folds.round(3).to_string(index=False))

    lstm = tabela.iloc[0]
    resumo = {"rmse": lstm["RMSE(log-vol)"], "r2_oos": lstm["R2_vs_media"],
              "ganho_vs_har_%": lstm["ganho_vs_HAR_%"], "qlike": lstm["QLIKE"]}
    return tabela, resumo, {"boot": df_boot, "folds": df_folds}


# ==============================================================================
# 7. GRÁFICOS, REGISTRO DE EXECUÇÕES E MODELO FINAL
# ==============================================================================
def _plot_loss(res, saida):
    h = res["hist"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["loss"], label="treino")
    ax.plot(h["val_loss"], label="validação")
    ax.set_title("Curva de loss (último modelo treinado)")
    ax.set_xlabel("época")
    ax.legend()
    fig.tight_layout()
    fig.savefig(saida / "loss.png", dpi=130)
    plt.close(fig)


def plotar_retorno(res, equities, cfg):
    saida = Path(cfg.saida)
    saida.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for nome in ("LSTM (ensemble)", "sempre_comprado", "ridge", "grad_boosting"):
        ax.plot(res["datas"], equities[nome], label=nome, lw=2.2 if nome.startswith("LSTM") else 1.1)
    ax.set_title(f"Backtest fora da amostra (custo {cfg.custo_bps:.0f} bps) — {cfg.ticker}")
    ax.set_ylabel("Capital (início = 1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(saida / "backtest.png", dpi=130)
    plt.close(fig)
    _plot_loss(res, saida)


def plotar_vol(res, cfg):
    saida = Path(cfg.saida)
    saida.mkdir(exist_ok=True)
    ens = np.mean(res["seeds"], axis=0)
    anual = lambda x: np.exp(x) * np.sqrt(252) * 100     # log-vol diária -> % ao ano
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(res["datas"], anual(res["y"]), color="lightgray", lw=1.5, label="realizada")
    ax.plot(res["datas"], anual(res["har"]), lw=1, label="HAR")
    ax.plot(res["datas"], anual(ens), lw=1.6, label="LSTM (ensemble)")
    ax.set_title(f"Volatilidade anualizada ({cfg.horizonte_vol}d à frente) — {cfg.ticker}")
    ax.set_ylabel("% ao ano")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(saida / "volatilidade.png", dpi=130)
    plt.close(fig)
    _plot_loss(res, saida)


def registrar_execucao(cfg, resumo):
    """Guarda cada execução. Quanto mais configurações você testa, maior a chance de um
    'resultado significativo' aparecer por acaso — o contador ajuda a corrigir isso."""
    saida = Path(cfg.saida)
    saida.mkdir(exist_ok=True)
    caminho = saida / "historico_execucoes.csv"
    linha = {"data_hora": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "ticker": cfg.ticker,
             "tarefa": cfg.tarefa, "horizonte": cfg.horizonte_vol if cfg.tarefa == "volatilidade" else cfg.horizonte,
             "lookback": cfg.lookback, "n_seeds": cfg.n_seeds, "custo_bps": cfg.custo_bps,
             "long_short": cfg.long_short, **{k: round(float(v), 4) for k, v in resumo.items()}}
    novo = pd.DataFrame([linha])
    hist = pd.concat([pd.read_csv(caminho), novo], ignore_index=True) if caminho.exists() else novo
    hist.to_csv(caminho, index=False)
    n = len(hist)
    print(f"\n[registro] execução nº {n} salva em {caminho}. Com {n} configuração(ões) testada(s), "
          f"use um limiar de p-valor mais rígido (≈ 0,05/{n} = {0.05 / n:.4f}).")


def treinar_e_salvar_final(df, cfg):
    feats = [c for c in df.columns if c not in ("alvo", "ret_seg")]
    X_raw, y = df[feats].values, df["alvo"].values
    n, L = len(df), cfg.lookback
    vol = cfg.tarefa == "volatilidade"
    corte, t_fit, t_val = dividir_treino(n, L, cfg.frac_validacao)
    scaler = StandardScaler().fit(X_raw[:corte])
    Xs = np.clip(scaler.transform(X_raw), -5, 5)
    mu = float(y[t_fit].mean()) if vol else 0.0
    sd = float(y[t_fit].std())
    modelo, _ = treinar_lstm(
        janelas(Xs, t_fit, L), alvo_lstm(y[t_fit], cfg, mu, sd),
        janelas(Xs, t_val, L), alvo_lstm(y[t_val], cfg, mu, sd), cfg, seed=cfg.seed)
    saida = Path(cfg.saida)
    saida.mkdir(exist_ok=True)
    modelo.save(saida / "lstm_final.keras")
    joblib.dump(scaler, saida / "scaler.joblib")
    (saida / "meta.json").write_text(json.dumps({
        "ticker": cfg.ticker, "features": feats, "lookback": L, "tarefa": cfg.tarefa,
        "horizonte": cfg.horizonte_vol if vol else cfg.horizonte,
        "alvo_media": mu, "alvo_desvio": sd, "treinado_ate": str(df.index[-1].date()),
    }, indent=2))
    print(f"Modelo final, scaler e metadados salvos em '{saida}/'.")


# ==============================================================================
# 8. PREVISÃO PARA O PRÓXIMO PREGÃO E ACOMPANHAMENTO DO QUE FOI PREVISTO
# ==============================================================================
def prever_amanha(cfg, progresso=None):
    """Treina com TODO o histórico disponível e prevê o(s) próximo(s) pregão(ões) após a última data.
    Direção: probabilidade de alta. Retorno: retorno esperado. Volatilidade: vol. esperada.
    ATENÇÃO: só faz sentido se a validação walk-forward mostrou que o modelo supera os baselines."""
    df = montar_features(cfg, incluir_futuro=True)
    feats = [c for c in df.columns if c not in ("alvo", "ret_seg")]
    X_raw, y = df[feats].values, df["alvo"].values
    n_total = len(df)
    n = int(np.sum(~np.isnan(y)))                      # linhas com alvo conhecido (as últimas não têm)
    if np.isnan(y[:n]).any():
        raise RuntimeError("Alvo com falhas no meio da série; verifique os dados.")
    vol, cls, L = cfg.tarefa == "volatilidade", cfg.tarefa == "classificacao", cfg.lookback
    h = cfg.horizonte_vol if vol else cfg.horizonte

    corte, t_fit, t_val = dividir_treino(n, L, cfg.frac_validacao)
    scaler = StandardScaler().fit(X_raw[:corte])
    Xs = np.clip(scaler.transform(X_raw), -5, 5)
    mu = float(y[t_fit].mean()) if vol else 0.0
    sd = float(y[t_fit].std())
    X_fit, X_val = janelas(Xs, t_fit, L), janelas(Xs, t_val, L)
    y_fit, y_val = alvo_lstm(y[t_fit], cfg, mu, sd), alvo_lstm(y[t_val], cfg, mu, sd)
    X_hoje = janelas(Xs, [n_total - 1], L)             # janela que termina na última data

    preds = []
    for s in range(cfg.n_seeds):
        modelo, _ = treinar_lstm(X_fit, y_fit, X_val, y_val, cfg, seed=cfg.seed + s)
        p = float(modelo.predict(X_hoje, verbose=0).ravel()[0])
        preds.append(p if cls else p * sd + mu)
        if progresso:
            progresso(s + 1, cfg.n_seeds, f"Modelo {s + 1}/{cfg.n_seeds} treinado")

    t_base, x_hoje = np.arange(L - 1, n), Xs[-1:]
    if vol:
        i = {c: feats.index(c) for c in ("lvol_5", "lvol_20", "lvol_60", "lvol_ewma", "lvol_pass")}
        cols = [i["lvol_5"], i["lvol_20"], i["lvol_60"]]
        har = LinearRegression().fit(Xs[t_base][:, cols], y[t_base])
        base = {"HAR": float(har.predict(x_hoje[:, cols])[0]), "EWMA": float(X_raw[-1, i["lvol_ewma"]]),
                "persistência": float(X_raw[-1, i["lvol_pass"]])}
    else:
        base = {"ridge": float(Ridge(alpha=cfg.ridge_alpha).fit(Xs[t_base], y[t_base]).predict(x_hoje)[0])}

    media = float(np.mean(preds))
    r = {"ticker": cfg.ticker, "tarefa": cfg.tarefa, "horizonte": h, "data_ref": str(df.index[-1].date()),
         "n_modelos": len(preds), "preds": preds, "media": media, "dispersao": float(np.std(preds)),
         "base": base, "n_treino": n}
    if cls:
        r["taxa_alta_hist"] = float(np.mean(y[:n] > 0))
    return r


PREV_COLUNAS = ["criado_em", "ticker", "data_ref", "tarefa", "horizonte", "previsto", "referencia", "realizado"]


def registrar_previsao(cfg, r):
    """Guarda a previsão em saida/previsoes.csv para conferir depois com o que realmente aconteceu."""
    saida = Path(cfg.saida)
    saida.mkdir(exist_ok=True)
    caminho = saida / "previsoes.csv"
    ref = r["taxa_alta_hist"] if r["tarefa"] == "classificacao" else next(iter(r["base"].values()))
    linha = pd.DataFrame([{"criado_em": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "ticker": r["ticker"],
                           "data_ref": r["data_ref"], "tarefa": r["tarefa"], "horizonte": r["horizonte"],
                           "previsto": r["media"], "referencia": ref, "realizado": np.nan}])
    antigo = pd.read_csv(caminho) if caminho.exists() else pd.DataFrame(columns=PREV_COLUNAS)
    pd.concat([antigo, linha], ignore_index=True)[PREV_COLUNAS].to_csv(caminho, index=False)


def atualizar_previsoes(cfg):
    """Preenche o que realmente aconteceu nas previsões cujo prazo já passou e calcula acertos."""
    caminho = Path(cfg.saida) / "previsoes.csv"
    if not caminho.exists():
        return pd.DataFrame(columns=PREV_COLUNAS)
    p = pd.read_csv(caminho)
    series = {}
    for i, row in p[p["realizado"].isna()].iterrows():
        if row["ticker"] not in series:
            series[row["ticker"]] = baixar(row["ticker"], cfg.inicio, None)
        close = series[row["ticker"]].get("Close")
        if close is None or close.empty:
            continue
        try:
            pos = close.index.get_loc(pd.Timestamp(row["data_ref"]))
        except KeyError:
            continue
        h = int(row["horizonte"])
        if pos + h >= len(close):
            continue                                       # o prazo ainda não passou
        if row["tarefa"] == "volatilidade":
            ret = np.log(close / close.shift(1)).values[pos + 1: pos + h + 1]
            p.loc[i, "realizado"] = float(np.log(max(np.sqrt(np.mean(ret ** 2)), 1e-6)))
        else:
            p.loc[i, "realizado"] = float(np.log(close.iloc[pos + h] / close.iloc[pos]))
    p.to_csv(caminho, index=False)
    ok = p["realizado"].notna()
    limiar = np.where(p["tarefa"] == "classificacao", 0.5, 0.0)
    p["acertou_direcao"] = np.where(ok & (p["tarefa"] != "volatilidade"),
                                    (p["previsto"] > limiar) == (p["realizado"] > 0), np.nan)
    return p


def interpretar_previsao(r):
    m = r["media"]
    pct = lambda x: (np.exp(x) - 1) * 100
    anual = lambda x: np.exp(x) * np.sqrt(252) * 100
    if r["tarefa"] == "classificacao":
        return {"prob_alta_%": 100 * m, "taxa_alta_historica_%": 100 * r["taxa_alta_hist"],
                "dispersao_entre_modelos_pp": 100 * r["dispersao"], "conviccao_%": 200 * abs(m - 0.5)}
    if r["tarefa"] == "regressao":
        return {"retorno_esperado_%": pct(m), "dispersao_entre_modelos_pp": 100 * r["dispersao"],
                **{f"{k}_retorno_%": pct(v) for k, v in r["base"].items()}}
    return {"vol_diaria_%": np.exp(m) * 100, "vol_anualizada_%": anual(m),
            **{f"{k}_vol_anualizada_%": anual(v) for k, v in r["base"].items()}}


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    cfg = Config()
    tf.keras.utils.set_random_seed(cfg.seed)
    df = montar_features(cfg)
    print(f"\n{cfg.ticker}: {len(df)} amostras | {df.shape[1] - 2} features | "
          f"{df.index[0].date()} → {df.index[-1].date()} | tarefa={cfg.tarefa}")

    res = executar_walk_forward(df, cfg)
    Path(cfg.saida).mkdir(exist_ok=True)
    if cfg.tarefa == "volatilidade":
        tabela, resumo, _ = relatorio_vol(res, cfg)
        plotar_vol(res, cfg)
    else:
        tabela, resumo, extras = relatorio_retorno(res, cfg)
        plotar_retorno(res, extras["equities"], cfg)
    tabela.to_csv(Path(cfg.saida) / f"resultados_{cfg.tarefa}.csv", index=False)
    registrar_execucao(cfg, resumo)

    if cfg.salvar_modelo_final:
        treinar_e_salvar_final(df, cfg)


if __name__ == "__main__":
    main()
