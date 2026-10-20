import shutil
import sys
import traceback
from dataclasses import replace
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import tensorflow as tf
import lstm_mercado_v2 as m


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def rodar(tarefa):
    cfg = replace(m.Config(), tarefa=tarefa)
    saida = Path(cfg.saida)
    saida.mkdir(exist_ok=True)
    with open(saida / f"console_{tarefa}.txt", "w", encoding="utf-8") as f:
        antigo = sys.stdout
        sys.stdout = Tee(antigo, f)
        try:
            tf.keras.utils.set_random_seed(cfg.seed)
            df = m.montar_features(cfg)
            print(f"\n{cfg.ticker}: {len(df)} amostras | {df.shape[1] - 2} features | "
                  f"{df.index[0].date()} -> {df.index[-1].date()} | tarefa={cfg.tarefa}")
            res = m.executar_walk_forward(df, cfg)
            if tarefa == "volatilidade":
                tabela, resumo, _ = m.relatorio_vol(res, cfg)
                m.plotar_vol(res, cfg)
            else:
                tabela, resumo, extras = m.relatorio_retorno(res, cfg)
                m.plotar_retorno(res, extras["equities"], cfg)
            tabela.to_csv(saida / f"resultados_{tarefa}.csv", index=False)
            m.registrar_execucao(cfg, resumo)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            raise
        finally:
            sys.stdout = antigo
    shutil.copy(saida / "loss.png", saida / f"loss_{tarefa}.png")


if __name__ == "__main__":
    for t in ("volatilidade", "classificacao"):
        rodar(t)
