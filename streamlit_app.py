import io
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

st.set_page_config(page_title="Simulador de Tanques", layout="wide")

SAIDAS = ["theta 600", "theta 300", "theta 200", "theta 100", "theta 6", "theta 3"]
SAIDAS_OBRIG = SAIDAS[:4]  # theta 6 e theta 3 só entram se existirem na planilha
MODELOS = {"Linear (extrapola)": "lin", "Polinomial grau 2 (extrapola)": "poly",
           "Random Forest": "rf", "Suave (Processo Gaussiano)": "gp"}
ENTRADAS = ["CMC", "Barita", "Sal"]
ARQ_PADRAO = "Dados_experimentais.xlsx"


# =====================================================================
# Modelo (planilha -> Random Forest)
# =====================================================================
class PredVisc:
    """Treina um RF: massas do lote (CMC, Barita, Sal em g) -> theta 600/300/200/100."""

    def __init__(self, fonte=None, tipo="rf"):
        self.rf, self.msg, self.lo, self.hi = None, "Sem planilha: θ indisponível", None, None
        self.tipo, self.saidas, self.escala, self.diag = tipo, list(SAIDAS_OBRIG), None, None
        if fonte is None and os.path.exists(ARQ_PADRAO):
            fonte = ARQ_PADRAO
        if fonte is None:
            return
        try:
            self._treinar(fonte)
        except Exception as e:  # planilha fora do formato esperado
            self.rf, self.msg = None, f"Erro ao ler planilha: {e}"

    @staticmethod
    def _achar(cols, chave):
        for c in cols:
            if chave.lower() in str(c).lower():
                return c
        return None

    @staticmethod
    def _achar_theta(cols, num):
        """Coluna cujo primeiro número é exatamente `num` (evita '6' casar com '600')."""
        for c in cols:
            m = re.search(r"\d+", str(c))
            if m and m.group() == num:
                return c
        return None

    def _treinar(self, fonte):
        for nome, df in pd.read_excel(fonte, sheet_name=None).items():
            ent = {e: self._achar(df.columns, e) for e in ENTRADAS}
            sai = {s: self._achar_theta(df.columns, s.split()[-1]) for s in SAIDAS}
            if ent["CMC"] and ent["Barita"] and all(sai[s] for s in SAIDAS_OBRIG):
                break
        else:
            raise ValueError("não achei colunas CMC, Barita e leituras 600/300/200/100")
        sai = {s: c for s, c in sai.items() if c}  # descarta 6 e 3 se ausentes
        X = pd.DataFrame({e: pd.to_numeric(df[c], errors="coerce") if c else 0.0 for e, c in ent.items()})
        Y = pd.DataFrame({s: pd.to_numeric(df[c], errors="coerce") for s, c in sai.items()})
        ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
        X, Y = X[ok], Y[ok]
        self.saidas = list(sai)
        self.diag = {
            "entradas": pd.DataFrame({
                "coluna usada": [str(ent[e]) if ent[e] else "(não achei: usando 0)" for e in ENTRADAS],
                "mín (g/lote)": X.min().values, "máx (g/lote)": X.max().values,
                "valores distintos": X.nunique().values}, index=ENTRADAS),
            "saidas": pd.DataFrame({
                "coluna usada": [str(c) for c in sai.values()],
                "mín": Y.min().values, "máx": Y.max().values,
                "valores distintos": Y.nunique().values}, index=list(sai)),
            "n": len(X)}
        if self.tipo == "gp":
            self.escala = StandardScaler().fit(X.values)
            ker = ConstantKernel(1.0) * RBF([1.0] * X.shape[1], (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-8, 1e-1))
            self.rf = GaussianProcessRegressor(ker, normalize_y=True, n_restarts_optimizer=3, random_state=0)
            self.rf.fit(self.escala.transform(X.values), Y.values)
            rotulo = "Processo Gaussiano"
        elif self.tipo == "lin":
            self.rf = make_pipeline(StandardScaler(), Ridge(alpha=0.1)).fit(X.values, Y.values)
            rotulo = "Linear"
        elif self.tipo == "poly":
            self.rf = make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=0.1)).fit(X.values, Y.values)
            rotulo = "Polinomial grau 2"
        else:
            self.rf = RandomForestRegressor(n_estimators=150, random_state=0).fit(X.values, Y.values)
            rotulo = "RF"
        self.lo, self.hi = X.min().values, X.max().values
        self.msg = f"{rotulo} treinado com {len(X)} experimentos (aba '{nome}'): " + ", ".join(
            s.split()[-1] for s in self.saidas)

    def prever(self, massas_lote):
        """massas_lote: dict {CMC, Barita, Sal} em g/lote. Retorna (dict theta, fora_da_faixa)."""
        # getattr protege contra instâncias antigas/incompletas mantidas em cache.
        rf = getattr(self, "rf", None)
        if rf is None:
            return None, {}
        x = np.array([[massas_lote.get(e, 0.0) for e in ENTRADAS]], dtype=float)
        y = rf.predict(self.escala.transform(x) if self.escala is not None else x)[0]
        y = np.clip(y, 0.0, None)  # leituras não são negativas

        # Identifica exatamente quais entradas estão fora do domínio experimental.
        # A comparação é feita na mesma unidade do treinamento: g por lote.
        fora = {
            e: {
                "valor": float(x[0, i]),
                "min": float(self.lo[i]),
                "max": float(self.hi[i]),
            }
            for i, e in enumerate(ENTRADAS)
            if x[0, i] < self.lo[i] - 1e-9 or x[0, i] > self.hi[i] + 1e-9
        }
        return dict(zip(self.saidas, y)), fora


@st.cache_resource(show_spinner="Treinando Random Forest…")
def carregar_modelo(dados, tipo="rf", versao_cache="modelo-v5-2026-11-09"):
    # A versão faz parte da chave do cache: força a criação de uma instância
    # nova quando esta implementação do modelo for publicada.
    return PredVisc(io.BytesIO(dados) if dados else None, tipo)


# =====================================================================
# Simulação (mesma lógica do HTML)
# =====================================================================
class Fluido:
    def __init__(self, rhos, m=None):
        self.rhos, self.m = rhos, dict(m or {})

    @property
    def volume(self):  # L
        return sum(v / self.rhos[e] for e, v in self.m.items() if e in self.rhos) / 1000

    @property
    def massa(self):
        return sum(self.m.values())

    @property
    def densidade(self):  # g/mL
        v = self.volume
        return self.massa / 1000 / v if v > 0 else 0.0

    def conc(self, e):  # g/L
        v = self.volume
        return self.m.get(e, 0.0) / v if v > 0 else 0.0

    def add(self, e, dm):
        self.m[e] = max(0.0, self.m.get(e, 0.0) + dm)

    def tirar(self, V):
        o, v = Fluido(self.rhos), self.volume
        if v <= 0 or V <= 0:
            return o
        f = min(1.0, V / v)
        for e in list(self.m):
            o.m[e] = self.m[e] * f
            self.m[e] -= o.m[e]
        return o

    def misturar(self, o):
        for e, v in o.m.items():
            self.m[e] = self.m.get(e, 0.0) + v


class Tanque:
    def __init__(self, id, nome, vol_max, agua, rhos, td=0, ruido=0.0):
        self.id, self.nome, self.vol_max, self.td, self.ruido = id, nome, float(vol_max), td, ruido
        self.fluido = Fluido(rhos, {"agua": float(agua)} if agua > 0 else {})
        self.overflow = Fluido(rhos)
        self.fora = {}
        self.H = {k: [] for k in ["t", "vol", "dens", "sensor", "conc", "theta", "extra"]}


class Valvula:
    def __init__(self, id, origem, destino, vazao, nome=""):
        self.id, self.origem, self.destino, self.vazao, self.nome, self.aberta = id, origem, destino, vazao, nome, False


class Bateria:
    def __init__(self, pred):
        self.pred, self.agua_lote = pred, 1000.0
        self.rhos = {"agua": 1.0, "CMC": 1.6, "Barita": 4.2, "Sal": 2.16}
        self.limpar()

    def limpar(self):
        self.tanques, self.valvulas, self.t, self._id, self.log = {}, {}, 0, 1, []

    def criar_tanque(self, nome, vmax, agua=0.0, td=0, ruido=0.0):
        t = Tanque(self._id, nome, vmax, agua, self.rhos, td, ruido)
        self._id += 1
        self.tanques[t.id] = t
        self._reg(t)
        return t

    def criar_valvula(self, nome, o, d, q):
        v = Valvula(self._id, o, d, q, nome)
        self._id += 1
        self.valvulas[v.id] = v
        return v

    def remover_tanque(self, id):
        self.tanques.pop(id, None)
        for v in [v for v in self.valvulas.values() if id in (v.origem, v.destino)]:
            del self.valvulas[v.id]

    def cenario(self):
        self.limpar()
        a = self.criar_tanque("T1", 5, 1000, 3, 1)
        b = self.criar_tanque("T2", 5, 10, 3, 1)
        c = self.criar_tanque("T3", 10, 10, 3, 1)
        self.criar_valvula("T1 → T2", a.id, b.id, 0.2)
        self.criar_valvula("T2 → T3", b.id, c.id, 0.1)

    def limpar_graficos(self):
        """Zera o histórico de todos os tanques e recomeça o gráfico do estado atual."""
        for t in self.tanques.values():
            t.H = {k: [] for k in ["t", "vol", "dens", "sensor", "conc", "theta", "extra"]}
            self._reg(t)

    def _reg(self, t):
        H, f = t.H, t.fluido
        H["t"].append(self.t)
        H["vol"].append(f.volume)
        H["dens"].append(f.densidade)
        d = H["dens"][max(0, len(H["dens"]) - 1 - t.td)]
        H["sensor"].append(d * (1 + (random.random() * 2 - 1) * t.ruido / 100) if d else 0.0)
        H["conc"].append({e: f.conc(e) for e in self.rhos})
        th, t.fora = None, {}
        agua = f.m.get("agua", 0.0)
        if agua > 1e-6:  # massa de cada espécie por lote da planilha (= agua_lote g de água)
            th, t.fora = self.pred.prever({e: f.m.get(e, 0.0) / agua * self.agua_lote for e in ENTRADAS})
        H["theta"].append(th)
        H["extra"].append(bool(t.fora))

    def passo(self, n=1):
        for _ in range(n):
            for v in self.valvulas.values():
                a, b = self.tanques.get(v.origem), self.tanques.get(v.destino)
                if v.aberta and a:
                    out = a.fluido.tirar(min(v.vazao, a.fluido.volume))
                    if b:
                        b.fluido.misturar(out)
            for t in self.tanques.values():
                V = t.fluido.volume
                if V > t.vol_max:
                    t.overflow.misturar(t.fluido.tirar(V - t.vol_max))
            self.t += 1
            for t in self.tanques.values():
                self._reg(t)

    def dosar(self, id, esp, m):
        self.tanques[id].fluido.add(esp, m)
        self.log.append(f"t={self.t}: dosou {m:g} g de {esp} em {self.tanques[id].nome}")
        self.passo(1)

    def transferir(self, id, dest, vol):
        a = self.tanques[id]
        out = a.fluido.tirar(vol)
        if dest is not None:
            self.tanques[dest].fluido.misturar(out)
        self.log.append(f"t={self.t}: transferiu {vol:g} L de {a.nome} para {'dreno' if dest is None else self.tanques[dest].nome}")
        self.passo(1)

    def csv(self):
        sp = list(self.rhos)
        cab = ["tanque", "t", "vol_L", "dens", "dens_sensor"] + [f"conc_{e}" for e in sp] + SAIDAS
        linhas = []
        for t in self.tanques.values():
            for i, ti in enumerate(t.H["t"]):
                th = t.H["theta"][i] or {}
                linhas.append([t.nome, ti, t.H["vol"][i], t.H["dens"][i], t.H["sensor"][i]]
                              + [t.H["conc"][i].get(e, 0.0) for e in sp] + [th.get(s, "") for s in SAIDAS])
        return pd.DataFrame(linhas, columns=cab).to_csv(index=False)


# =====================================================================
# Estado
# =====================================================================
up = st.sidebar.file_uploader("Planilha de dados experimentais (.xlsx)", type="xlsx")
tipo_modelo = MODELOS[st.sidebar.selectbox(
    "Modelo dos θ", list(MODELOS),
    help="Linear e Polinomial: contínuos e continuam variando fora da faixa (extrapolam). "
         "Random Forest: responde em degraus (a planilha tem só 3 níveis por variável) e repete o valor "
         "da borda fora da faixa. Processo Gaussiano: contínuo, mas foi o pior na validação com a planilha.")]
pred = carregar_modelo(up.getvalue() if up else None, tipo_modelo)
VERSAO_ESTADO = 5
if st.session_state.get("versao_estado") != VERSAO_ESTADO:
    # Objetos Bateria/Tanque salvos por versões antigas podem ter atributos incompatíveis.
    for _k in ("bat", "gen", "run", "csv"):
        st.session_state.pop(_k, None)
    st.session_state.versao_estado = VERSAO_ESTADO
if "bat" not in st.session_state:
    st.session_state.bat, st.session_state.gen, st.session_state.run = Bateria(pred), 0, False
    st.session_state.bat.cenario()
    st.session_state.csv = ""
bat, g = st.session_state.bat, st.session_state.gen
bat.pred = pred
st.sidebar.caption(pred.msg)
if pred.diag:
    with st.sidebar.expander("🔎 Diagnóstico do modelo"):
        st.caption(f"{pred.diag['n']} experimentos. Entradas do modelo = massa de cada espécie por lote de água da planilha.")
        st.dataframe(pred.diag["entradas"], use_container_width=True)
        st.dataframe(pred.diag["saidas"], use_container_width=True)
bat.agua_lote = st.sidebar.number_input(
    "Água por lote da planilha (g)", 1.0, value=float(bat.agua_lote), step=50.0,
    help="As massas da planilha (CMC, Barita, Sal) são por lote. Pela densidade das linhas, o lote tem ~1000 mL "
         "de água. O app converte cada tanque para essa base: massa da espécie ÷ massa de água × este valor.")
nm = lambda i: "dreno" if i is None else bat.tanques[i].nome


def recarregar(cenario):
    st.session_state.gen += 1  # novas chaves de widgets
    st.session_state.run = False
    bat.cenario() if cenario else bat.limpar()


# =====================================================================
# Barra superior
# =====================================================================
st.title("🧪 Simulador de tanques")
c = st.columns([1, 1, 2, 2, 1.4, 1])
if c[0].button("Passo ⏭"):
    bat.passo(1)
if c[1].button("⏸ Pausar" if st.session_state.run else "▶ Rodar", type="primary"):
    st.session_state.run = not st.session_state.run
    st.rerun()
vel = c[2].slider("velocidade (ms/passo)", 100, 2000, 500, 50)
if c[3].button("Carregar cenário T1→T2→T3"):
    recarregar(True)
    st.rerun()
if c[4].button("Limpar tudo"):
    recarregar(False)
    st.rerun()

# ---------- Espécies ----------
st.subheader("Espécies (o fluido é montado a partir delas)")
cols = st.columns(max(5, len(bat.rhos) + 1))
for col, e in zip(cols, list(bat.rhos)):
    bat.rhos[e] = col.number_input(f"{e} (g/mL)", 0.01, value=float(bat.rhos[e]), step=0.01, key=f"{g}_rho_{e}")
with st.form("nova_esp", clear_on_submit=True, border=False):
    f1, f2, f3 = st.columns([2, 2, 1])
    n_esp, r_esp = f1.text_input("Nova espécie", placeholder="nome"), f2.number_input("ρ (g/mL)", 0.01, value=1.0)
    f3.write("")
    if f3.form_submit_button("+ espécie") and n_esp.strip():
        bat.rhos[n_esp.strip()] = r_esp
        st.rerun()


# =====================================================================
# Painel dinâmico (reexecuta sozinho quando "Rodar" está ativo)
# =====================================================================
def card(t):
    k = lambda s: f"{g}_{s}{t.id}"
    f = t.fluido
    h = st.columns([3, 2, 1])
    h[0].markdown(f"**{t.nome}**")
    h[1].caption(f"máx {t.vol_max:g} L")
    if h[2].button("✕", key=k("rm")):
        bat.remover_tanque(t.id)
        st.rerun()
    nivel = min(100, f.volume / t.vol_max * 100)
    comp = "<br>".join(f"{e}: {m:.2f} g ({f.conc(e):.2f} g/L)" for e, m in f.m.items() if m > 1e-9) or "vazio"
    ov = f"<br><span style='color:#d64545'>overflow: {t.overflow.volume:.3f} L</span>" if t.overflow.volume > 0 else ""
    st.markdown(
        f"<div style='display:flex;gap:10px;margin:4px 0'>"
        f"<div style='position:relative;width:44px;height:96px;border:2px solid #888;border-top:none;"
        f"border-radius:0 0 8px 8px;overflow:hidden;flex:none'>"
        f"<div style='position:absolute;bottom:0;left:0;right:0;height:{nivel}%;background:#2f6fed;opacity:.65'></div></div>"
        f"<div style='font-size:12px'><b>{f.volume:.4f} L</b> · ρ {f.densidade:.4f} g/mL<br>{comp}{ov}</div></div>",
        unsafe_allow_html=True)
    th = t.H["theta"][-1] if t.H["theta"] else None
    if th:
        st.caption(" · ".join(f"θ{s.split()[-1]} {th[s]:.1f}" for s in SAIDAS if s in th)
                   + f"  |  VA {th['theta 600']/2:.1f} · VP {th['theta 600']-th['theta 300']:.1f}"
                   f" · LE {th['theta 300']-(th['theta 600']-th['theta 300']):.1f}")
        fora = t.fora if isinstance(t.fora, dict) else {}
        if fora:
            detalhes = " | ".join(
                f"{e}: {v['valor']:.3f} g/lote (treino: {v['min']:.3f} a {v['max']:.3f})"
                for e, v in fora.items()
            )
            dica = (" O Random Forest repete o valor da borda fora da faixa; "
                    "os modelos Linear e Polinomial (barra lateral) extrapolam." if pred.tipo == "rf" else "")
            st.warning("⚠️ Extrapolando: a previsão foi feita, mas há entrada(s) fora da faixa de treino. "
                       + detalhes + f" (g por lote de {bat.agua_lote:g} g de água = massa da espécie ÷ massa de água do tanque × {bat.agua_lote:g})."
                       + dica)
    elif f.volume > 0 and getattr(pred, "rf", None) is not None:
        st.caption("θ indisponível: tanque sem água (a planilha define a composição por massa de água do lote).")
    a, b, d = st.columns([2, 2, 1.2])
    esp = a.selectbox("Espécie", list(bat.rhos), key=k("e"), label_visibility="collapsed")
    m = b.number_input("g", value=1.0, step=0.5, key=k("m"), label_visibility="collapsed")
    if d.button("Dosar", key=k("d")):
        bat.dosar(t.id, esp, m)
        st.rerun()
    a, b, d = st.columns([1.6, 2, 1.2])
    vol = a.number_input("L", 0.0, value=0.5, step=0.1, key=k("tv"), label_visibility="collapsed")
    dest = b.selectbox("→", [None] + [i for i in bat.tanques if i != t.id], format_func=nm, key=k("dt"),
                       label_visibility="collapsed")
    if d.button("Ir", key=k("tr")):
        bat.transferir(t.id, dest, vol)
        st.rerun()
    a, b = st.columns(2)
    t.td = int(a.number_input("sensor: atraso", 0, value=t.td, key=k("sd")))
    t.ruido = b.number_input("ruído %", 0.0, value=float(t.ruido), step=0.5, key=k("sr"))


def painel():
    if st.session_state.run:
        bat.passo(1)
    st.caption(f"t = {bat.t}")

    # ---------- Tanques ----------
    st.subheader("Tanques")
    with st.expander("➕ Novo tanque"), st.form("novo_tanque", clear_on_submit=True, border=False):
        f1, f2, f3, f4 = st.columns(4)
        nn = f1.text_input("Nome", placeholder="nome")
        nv = f2.number_input("Vol. máx (L)", 0.1, value=5.0)
        na = f3.number_input("Água inicial (g)", 0.0, value=0.0)
        f4.write("")
        if f4.form_submit_button("+ tanque", type="primary"):
            bat.criar_tanque(nn or f"T{bat._id}", nv, na)
            st.rerun()
    tanques = list(bat.tanques.values())
    for i in range(0, len(tanques), 3):
        for col, t in zip(st.columns(3), tanques[i:i + 3]):
            with col, st.container(border=True):
                card(t)

    # ---------- Válvulas ----------
    st.subheader("Válvulas (transferência contínua, L por passo)")
    if tanques:
        with st.form("nova_valvula", clear_on_submit=True, border=False):
            g1, g2, g3, g4 = st.columns([2, 2, 1.5, 1])
            vo = g1.selectbox("Origem", list(bat.tanques), format_func=nm)
            vd = g2.selectbox("Destino", [None] + list(bat.tanques), format_func=nm)
            vq = g3.number_input("L/passo", 0.0, value=0.2, step=0.05)
            g4.write("")
            if g4.form_submit_button("+ válvula", type="primary") and vo != vd:
                bat.criar_valvula(f"{nm(vo)} → {nm(vd)}", vo, vd, vq)
                st.rerun()
    for v in list(bat.valvulas.values()):
        with st.container(border=True):
            c = st.columns([3, 1.5, 1.5, 0.8])
            c[0].markdown(f"**{nm(v.origem)} → {nm(v.destino)}**")
            v.vazao = c[1].number_input("L/passo", 0.0, value=float(v.vazao), step=0.05, key=f"{g}_vq{v.id}",
                                        label_visibility="collapsed")
            v.aberta = c[2].toggle("Aberta", v.aberta, key=f"{g}_va{v.id}")
            if c[3].button("✕", key=f"{g}_vx{v.id}"):
                del bat.valvulas[v.id]
                st.rerun()

    # ---------- Gráficos ----------
    st.subheader("Gráficos")
    tem_modelo_rf = getattr(pred, "rf", None) is not None
    ops = (
        ["Volume (L)", "Densidade real + sensor (g/mL)"]
        + (list(pred.saidas) if tem_modelo_rf else [])
        + [f"Conc. {e} (g/L)" for e in bat.rhos]
    )
    sel, btn = st.columns([4, 1])
    met = sel.selectbox("Variável (eixo x = tempo global, passos)", ops)
    btn.write("")
    btn.write("")
    if btn.button("🧹 Limpar gráfico", help="Apaga o histórico plotado e recomeça do estado atual dos tanques."):
        bat.limpar_graficos()
    if tanques:
        fig, ax = plt.subplots(figsize=(10, 3.8))
        for i, t in enumerate(tanques):
            H, x, cor = t.H, t.H["t"], f"C{i}"
            if met == "Volume (L)":
                ax.step(x, H["vol"], where="post", label=t.nome, color=cor)
            elif met.startswith("Densidade"):
                ax.step(x, H["dens"], where="post", label=f"{t.nome} real", color=cor)
                ax.step(x, H["sensor"], where="post", label=f"{t.nome} sensor", color=cor, ls="--", alpha=0.6)
            elif met in SAIDAS:
                y = [h[met] if h else np.nan for h in H["theta"]]
                ax.step(x, y, where="post", label=t.nome, color=cor)
                ex = [j for j, e in enumerate(H["extra"]) if e and j < len(y)]
                if ex:
                    ax.scatter([x[j] for j in ex], [y[j] for j in ex], marker="x", color=cor, zorder=3,
                               label=f"{t.nome} (extrapolando)")
            else:
                e = met[6:-6]
                ax.step(x, [cc.get(e, 0.0) for cc in H["conc"]], where="post", label=t.nome, color=cor)
        ax.set_xlabel("tempo (passos)")
        ax.set_ylabel(met)
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # ---------- CSV ----------
    c = st.columns([1, 1, 4])
    if c[0].button("Gerar CSV"):
        st.session_state.csv = bat.csv()
    c[1].download_button("⬇ Baixar CSV", bat.csv(), "simulacao.csv", "text/csv")
    st.text_area("CSV", st.session_state.csv, height=120, placeholder="o CSV aparece aqui para copiar",
                 label_visibility="collapsed")
    with st.expander("Registro de operações"):
        st.text("\n".join(reversed(bat.log)))


# run_every dinâmico: só reexecuta sozinho quando "Rodar" está ativo
st.fragment(painel, run_every=vel / 1000 if st.session_state.run else None)()
