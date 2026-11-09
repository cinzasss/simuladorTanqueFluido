import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from simulador import SAIDAS, Bateria, PredVisc

st.set_page_config(page_title="Simulador de Tanques", page_icon="🧪", layout="wide")


@st.cache_resource(show_spinner="Treinando Random Forest…")
def carregar_modelo(dados):
    return PredVisc(io.BytesIO(dados) if dados else None)


# ---------- Estado e modelo ----------
up = st.sidebar.file_uploader("Excel de dados experimentais (opcional)", type="xlsx")
pred = carregar_modelo(up.getvalue() if up else None)
if "bat" not in st.session_state:
    st.session_state.bat, st.session_state.gen = Bateria(pred), 0
    st.session_state.bat.cenario()
bat, g = st.session_state.bat, st.session_state.gen
bat.pred = pred
st.sidebar.caption(pred.msg)
nm = lambda i: "dreno" if i is None else bat.tanques[i].nome


def recarregar(cenario):
    st.session_state.gen += 1  # novas chaves de widgets, evita valores antigos
    bat.cenario() if cenario else bat.limpar()
    st.rerun()


# ---------- Espécies (sidebar) ----------
st.sidebar.subheader("Espécies (ρ em g/mL)")
for e in list(bat.rhos):
    bat.rhos[e] = st.sidebar.number_input(e, 0.01, value=float(bat.rhos[e]), step=0.01, key=f"rho_{e}")
with st.sidebar.form("nova_esp", clear_on_submit=True):
    n_esp, r_esp = st.text_input("Nova espécie"), st.number_input("ρ (g/mL)", 0.01, value=1.0)
    if st.form_submit_button("Adicionar") and n_esp.strip():
        bat.rhos[n_esp.strip()] = r_esp
        st.rerun()

# ---------- Controle de tempo ----------
st.title("🧪 Simulador de tanques")
c = st.columns([1.2, 1, 1, 1.4, 1])
n = c[0].number_input("Passos", 1, 1000, 10)
if c[1].button("▶ Rodar n passos"):
    bat.passo(int(n))
if c[2].button("⏭ 1 passo"):
    bat.passo(1)
if c[3].button("Cenário T1→T2→T3"):
    recarregar(True)
if c[4].button("Limpar tudo"):
    recarregar(False)
st.caption(f"t = {bat.t}")


# ---------- Tanques ----------
def card(t):
    k = lambda s: f"{g}_{s}{t.id}"
    f = t.fluido
    t.nome = st.text_input("Tanque", t.nome, key=k("n"))
    f.nome = st.text_input("Fluido", f.nome, key=k("f"))
    t.vol_max = st.number_input("Volume máx (L)", 0.1, value=float(t.vol_max), step=0.5, key=k("v"))
    st.progress(min(1.0, f.volume / t.vol_max), text=f"{f.volume:.4f} L / {t.vol_max:g} L")
    a, b = st.columns(2)
    a.metric("Densidade (g/mL)", f"{f.densidade:.4f}")
    b.metric("Sensor (g/mL)", f"{t.H['sensor'][-1]:.4f}" if t.H["sensor"] else "–")
    th = t.H["theta"][-1] if t.H["theta"] else None
    if th:
        for col, cn in zip(st.columns(len(SAIDAS)), SAIDAS):
            col.metric(cn.replace("theta ", "θ"), f"{th[cn]:.1f}")
        a6, a3 = th.get("theta 600"), th.get("theta 300")
        if a6 is not None and a3 is not None:
            st.caption(f"VA {a6/2:.1f} · VP {a6-a3:.1f} · LE {a3-(a6-a3):.1f}")
        if t.fora:
            st.warning("Concentração fora da faixa de treino do RF")
    elif f.volume > 0:
        st.caption("θ indisponível (sem modelo)")
    comp = {e: {"Massa (g)": m, "Conc. (g/L)": f.concentracao(e)} for e, m in f.massas.items() if m > 1e-9}
    if comp:
        st.dataframe(pd.DataFrame(comp).T.round(3), use_container_width=True)
    if t.overflow.volume > 0:
        st.error(f"Overflow acumulado: {t.overflow.volume:.3f} L")
    with st.expander("Operações"):
        esp = st.selectbox("Espécie", list(bat.rhos), key=k("e"))
        m = st.number_input("Massa (g)", value=1.0, step=0.5, key=k("m"))
        if st.button("Dosar", key=k("d")):
            bat.dosar(t.id, esp, m)
            st.rerun()
        vol = st.number_input("Volume (L)", 0.0, value=0.5, step=0.1, key=k("tv"))
        dest = st.selectbox("Destino", [None] + [i for i in bat.tanques if i != t.id], format_func=nm, key=k("dt"))
        if st.button("Transferir", key=k("tr")):
            bat.transferir(t.id, dest, vol)
            st.rerun()
    with st.expander("Sensor de densidade"):
        t.td = int(st.number_input("Atraso (passos)", 0, value=t.td, key=k("sd")))
        t.ruido = st.number_input("Ruído (%)", 0.0, value=t.ruido * 100, step=0.5, key=k("sr")) / 100
    if st.button("Remover tanque", key=k("rm")):
        bat.remover_tanque(t.id)
        st.rerun()


st.subheader("Tanques")
tanques = list(bat.tanques.values())
for i in range(0, len(tanques), 3):
    for col, t in zip(st.columns(3), tanques[i:i + 3]):
        with col, st.container(border=True):
            card(t)
with st.expander("➕ Novo tanque"), st.form("novo_tanque", clear_on_submit=True):
    f1, f2, f3, f4 = st.columns(4)
    nn, nf = f1.text_input("Nome do tanque"), f2.text_input("Nome do fluido")
    nv, na = f3.number_input("Volume máx (L)", 0.1, value=5.0), f4.number_input("Água inicial (g)", 0.0, value=0.0)
    if st.form_submit_button("Criar"):
        bat.criar_tanque(nn or f"Tanque {bat._id}", nv, na, nf or "Fluido")
        st.rerun()

# ---------- Válvulas ----------
st.subheader("Válvulas")
for v in list(bat.valvulas.values()):
    c = st.columns([3, 3, 1.5, 1.2, 0.8])
    v.nome = c[0].text_input("Nome", v.nome, key=f"{g}_vn{v.id}", label_visibility="collapsed")
    c[1].write(f"{nm(v.origem)} → {nm(v.destino)}")
    v.vazao = c[2].number_input("L/passo", 0.0, value=float(v.vazao), step=0.05, key=f"{g}_vq{v.id}", label_visibility="collapsed")
    v.aberta = c[3].toggle("Aberta", v.aberta, key=f"{g}_va{v.id}")
    if c[4].button("✕", key=f"{g}_vx{v.id}"):
        del bat.valvulas[v.id]
        st.rerun()
if len(bat.tanques) >= 1:
    with st.expander("➕ Nova válvula"), st.form("nova_valvula", clear_on_submit=True):
        g1, g2, g3, g4 = st.columns(4)
        vn = g1.text_input("Nome da válvula")
        vo = g2.selectbox("Origem", list(bat.tanques), format_func=nm)
        vd = g3.selectbox("Destino", [None] + list(bat.tanques), format_func=nm)
        vq = g4.number_input("Vazão (L/passo)", 0.0, value=0.2, step=0.05)
        if st.form_submit_button("Criar") and vo != vd:
            bat.criar_valvula(vn or f"{nm(vo)} → {nm(vd)}", vo, vd, vq)
            st.rerun()

# ---------- Gráficos ----------
st.subheader("Gráficos")
ops = ["Volume (L)", "Densidade real + sensor (g/mL)"] + SAIDAS + [f"Conc. {e} (g/L)" for e in bat.rhos]
met = st.selectbox("Variável", ops)
if tanques:
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, t in enumerate(tanques):
        H, x, cor = t.H, t.H["t"], f"C{i}"
        if met == "Volume (L)":
            ax.step(x, H["vol"], where="post", label=t.nome, color=cor)
        elif met.startswith("Densidade"):
            ax.step(x, H["dens"], where="post", label=f"{t.nome} real", color=cor)
            ax.step(x, H["sensor"], where="post", label=f"{t.nome} sensor", color=cor, ls="--", alpha=0.6)
        elif met in SAIDAS:
            ax.step(x, [h[met] if h else np.nan for h in H["theta"]], where="post", label=t.nome, color=cor)
        else:
            e = met[6:-6]
            ax.step(x, [cc.get(e, 0.0) for cc in H["conc"]], where="post", label=t.nome, color=cor)
    ax.set_xlabel("tempo (passos)"); ax.set_ylabel(met); ax.grid(True); ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with st.expander("Registro de operações"):
    st.text("\n".join(reversed(bat.log)))
st.download_button("⬇ Baixar CSV", bat.csv(), "simulacao.csv", "text/csv")
