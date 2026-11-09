"""Núcleo do simulador (sem interface): Fluido, PredVisc, Tanque, Valvula e Bateria."""
import csv
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ---------------- Configuração ----------------
XLSX = Path(os.getenv("DADOS_XLSX", "dados/Dados experimentais.xlsx"))
ABA = os.getenv("DADOS_ABA", "Resultados")
ENTRADAS = ["CMC", "Barita", "Sal"]  # espécies que alimentam o RF (g/L, mesma unidade do Excel)
SAIDAS = ["theta 600", "theta 300", "theta 200", "theta 100"]
RHOS_PADRAO = {"agua": 1.0, "CMC": 1.6, "Barita": 4.2, "Sal": 2.16}  # g/mL
MAX_HIST = 3000


# ---------------- Fluido (entidade independente do tanque) ----------------
class Fluido:
    def __init__(self, rhos, massas=None, nome="Fluido"):
        self.rhos = rhos  # referência compartilhada: editar a densidade de uma espécie vale para todos
        self.massas = dict(massas or {})
        self.nome = nome

    @property
    def volume(self):  # L (massa em g, rho em g/mL)
        return sum(m / self.rhos[e] for e, m in self.massas.items() if e in self.rhos) / 1000.0

    @property
    def densidade(self):  # g/mL
        v = self.volume
        return (sum(self.massas.values()) / 1000.0) / v if v > 0 else 0.0

    def concentracao(self, especie):  # g/L
        v = self.volume
        return self.massas.get(especie, 0.0) / v if v > 0 else 0.0

    def adicionar(self, especie, dm):
        if especie not in self.rhos:
            raise ValueError(f"Espécie desconhecida: {especie}")
        self.massas[especie] = max(0.0, self.massas.get(especie, 0.0) + dm)

    def retirar(self, volume_L):
        v = self.volume
        saida = Fluido(self.rhos, {}, self.nome)
        if v <= 0 or volume_L <= 0:
            return saida
        frac = min(1.0, volume_L / v)
        for e, m in self.massas.items():
            saida.massas[e] = m * frac
            self.massas[e] = m - saida.massas[e]
        return saida

    def misturar_com(self, outro):
        for e, m in outro.massas.items():
            self.massas[e] = self.massas.get(e, 0.0) + m


# ---------------- Modelo de θ ----------------
class PredVisc:
    def __init__(self, fonte=None):  # fonte: caminho ou arquivo enviado (upload); padrão = XLSX
        fonte = fonte or XLSX
        nome = fonte.name if isinstance(fonte, Path) else "arquivo enviado"
        self.modelo, self.lim = None, None
        try:
            df = pd.read_excel(fonte, sheet_name=ABA).dropna(subset=ENTRADAS + SAIDAS)
            X, y = df[ENTRADAS].values.astype(float), df[SAIDAS].values.astype(float)
            self.modelo = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1).fit(X, y)
            self.modelo.set_params(n_jobs=1)  # previsões pequenas e frequentes: sem overhead de paralelismo
            self.lim = (X.min(0), X.max(0))
            self.msg = f"Random Forest treinado com {len(df)} amostras ({nome})"
        except Exception as e:  # sem Excel o app roda, só sem θ
            self.msg = f"Sem modelo de θ: {e}"

    def prever(self, X):
        if self.modelo is None or not X:
            return [None] * len(X)
        return [dict(zip(SAIDAS, map(float, p))) for p in self.modelo.predict(np.array(X))]

    def fora_do_dominio(self, x):
        return self.lim is not None and bool(np.any(np.array(x) < self.lim[0]) or np.any(np.array(x) > self.lim[1]))


# ---------------- Tanque, Válvula, Bateria ----------------
class Tanque:
    def __init__(self, id, nome, vol_max, fluido, td=0, ruido=0.0):
        self.id, self.nome, self.vol_max, self.fluido = id, nome, float(vol_max), fluido
        self.td, self.ruido = int(td), float(ruido)  # atraso (passos) e ruído (fração) do sensor
        self.overflow = Fluido(fluido.rhos, {}, "Overflow")
        self.fora = False
        self.H = {k: [] for k in ("t", "vol", "dens", "sensor", "conc", "theta")}

    def registrar(self, t, theta):
        H, f = self.H, self.fluido
        H["t"].append(t); H["vol"].append(f.volume); H["dens"].append(f.densidade)
        m = H["dens"][max(0, len(H["dens"]) - 1 - self.td)]
        H["sensor"].append(m * (1 + np.random.uniform(-self.ruido, self.ruido)) if m else 0.0)
        H["conc"].append({e: f.concentracao(e) for e in f.rhos})
        H["theta"].append(theta)
        for k in H:
            del H[k][:-MAX_HIST]


class Valvula:
    def __init__(self, id, nome, origem, destino, vazao):
        self.id, self.nome, self.origem, self.destino = id, nome, origem, destino  # destino None = dreno
        self.vazao, self.aberta = float(vazao), False  # L por passo


class Bateria:
    def __init__(self, pred):
        self.pred, self.rhos = pred, dict(RHOS_PADRAO)
        self.limpar()

    def limpar(self):
        self.tanques, self.valvulas, self.t, self.log, self._id = {}, {}, 0, [], 1

    def _novo_id(self):
        self._id += 1
        return self._id - 1

    def _log(self, msg):
        self.log.append(f"t={self.t} · {msg}")
        del self.log[:-100]

    def _registrar(self, tanques):
        X = [[t.fluido.concentracao(e) for e in ENTRADAS] for t in tanques]
        for t, x, th in zip(tanques, X, self.pred.prever(X)):
            vazio = t.fluido.volume <= 0
            t.fora = (not vazio) and self.pred.fora_do_dominio(x)
            t.registrar(self.t, None if vazio else th)

    def criar_tanque(self, nome, vol_max, agua_g=0, fluido_nome="Fluido", td=0, ruido=0.0):
        f = Fluido(self.rhos, {"agua": float(agua_g)} if float(agua_g) > 0 else {}, fluido_nome)
        t = Tanque(self._novo_id(), nome, vol_max, f, td, ruido)
        self.tanques[t.id] = t
        self._registrar([t])
        self._log(f"Tanque “{nome}” criado")
        return t

    def remover_tanque(self, id):
        self.tanques.pop(id)
        self.valvulas = {k: v for k, v in self.valvulas.items() if id not in (v.origem, v.destino)}

    def criar_valvula(self, nome, origem, destino, vazao):
        v = Valvula(self._novo_id(), nome, origem, destino, vazao)
        self.valvulas[v.id] = v
        return v

    def dosar(self, id, especie, massa_g):
        t = self.tanques[id]
        t.fluido.adicionar(especie, massa_g)
        self._log(f"Dosagem de {massa_g:g} g de {especie} em “{t.nome}”")
        self.passo()

    def transferir(self, id, destino, volume_L):
        o = self.tanques[id]
        d = self.tanques.get(destino) if destino else None
        saida = o.fluido.retirar(volume_L)
        if d:
            d.fluido.misturar_com(saida)
        self._log(f"Transferência de {volume_L:g} L de “{o.nome}” para “{d.nome if d else 'dreno'}”")
        self.passo()

    def passo(self, n=1):
        for _ in range(n):
            for v in self.valvulas.values():
                if v.aberta:
                    o = self.tanques[v.origem]
                    saida = o.fluido.retirar(min(v.vazao, o.fluido.volume))
                    if v.destino:
                        self.tanques[v.destino].fluido.misturar_com(saida)
            for t in self.tanques.values():
                excesso = t.fluido.volume - t.vol_max
                if excesso > 0:
                    t.overflow.misturar_com(t.fluido.retirar(excesso))
            self.t += 1
            self._registrar(list(self.tanques.values()))

    def cenario(self):
        self.limpar()
        t1 = self.criar_tanque("T1 – Alimentação", 5, 1000, "Lama base", 3, 0.01)
        t2 = self.criar_tanque("T2 – Mistura", 5, 10, "Fluido de mistura", 3, 0.01)
        t3 = self.criar_tanque("T3 – Produto", 10, 10, "Fluido produto", 3, 0.01)
        self.criar_valvula("V1 – Alimentação → Mistura", t1.id, t2.id, 0.2)
        self.criar_valvula("V2 – Mistura → Produto", t2.id, t3.id, 0.1)

    def estado(self):
        def tanque(t):
            f, H = t.fluido, t.H
            return {
                "id": t.id, "nome": t.nome, "fluido": f.nome, "vol_max": t.vol_max, "volume": f.volume,
                "densidade": f.densidade, "massas": f.massas, "conc": {e: f.concentracao(e) for e in f.rhos},
                "theta": H["theta"][-1] if H["theta"] else None, "fora_dominio": t.fora,
                "overflow_L": t.overflow.volume, "sensor_td": t.td, "sensor_ruido_pct": t.ruido * 100,
                "hist": {
                    "t": H["t"], "vol": H["vol"], "dens": H["dens"], "sensor": H["sensor"],
                    "theta": {c: [h[c] if h else None for h in H["theta"]] for c in SAIDAS},
                    "conc": {e: [c.get(e, 0.0) for c in H["conc"]] for e in self.rhos},
                },
            }
        return {
            "t": self.t, "especies": self.rhos, "modelo": self.pred.msg, "saidas": SAIDAS,
            "tanques": [tanque(t) for t in self.tanques.values()], "log": self.log[-40:],
            "valvulas": [{"id": v.id, "nome": v.nome, "origem": v.origem, "destino": v.destino,
                          "vazao": v.vazao, "aberta": v.aberta} for v in self.valvulas.values()],
        }

    def csv(self):
        out = io.StringIO()
        w = csv.writer(out)
        esp = list(self.rhos)
        w.writerow(["tanque", "fluido", "t", "vol_L", "dens", "dens_sensor"] + [f"conc_{e}" for e in esp] + SAIDAS)
        for t in self.tanques.values():
            for i, ti in enumerate(t.H["t"]):
                th = t.H["theta"][i] or {}
                w.writerow([t.nome, t.fluido.nome, ti, t.H["vol"][i], t.H["dens"][i], t.H["sensor"][i]]
                           + [t.H["conc"][i].get(e, 0.0) for e in esp] + [th.get(c, "") for c in SAIDAS])
        return out.getvalue()
