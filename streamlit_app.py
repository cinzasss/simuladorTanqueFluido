import io
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from simulador import SAIDAS, Bateria, PredVisc

st.set_page_config(page_title="Simulador de Tanques", page_icon="🧪", layout="wide")
st.markdown("""<style>
.tq{width:56px;height:125px;border:2px solid currentColor;border-top:none;border-radius:0 0 10px 10px;position:relative;overflow:hidden}
.tq .lv{position:absolute;bottom:0;left:0;right:0;background:#2f6fed;opacity:.65;transition:height .3s}
.tq.over{border-color:#d64545}
.inf{font-size:13px;line-height:1.45;overflow-wrap:anywhere}
.mut{opacity:.6}.warn{color:#d64545}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#d64545;margin-right:6px}.dot.on{background:#2a9d57}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Treinando Random Forest…")
def carregar_modelo(dados):
    return PredVisc(io.BytesIO(dados) if dados else None)


# ---------- Estado e modelo ----------
up = st.sidebar.file_uploader("Excel de dados experimentais (opcional)", type="xlsx")
pred = carregar_modelo(up.getvalue() if up else None)
if "bat" not in st.session_state:
    st.session_state.bat, st.session_state.gen = Bateria(pred), 0
    st.session_state.bat.cenario()
    st.session_state.rodando, st.session_state.ult = False, 0.0
bat, g = st.session_state.bat, st.session_state.gen
bat.pred = pred
st.sidebar.caption(pred.msg)
nm = lambda i: "dreno" if i is None else bat.tanques[i].nome


def recarregar(cenario):
    st.session_state.gen += 1  # novas chaves de widgets, evita valores antigos
    st.session_state.rodando = False
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
rodando = st.session_state.rodando
c = st.columns([1, 1, 1, 1.2, 1.2, 1.3, 1])
n = c[0].number_input("Passos", 1, 1000, 10)
if c[1].button("Rodar n passos"):
    bat.passo(int(n))
if c[2].button("⏭ 1 passo"):
    bat.passo(1)
if c[3].button("⏸ Pausar" if rodando else "▶ Play", type="primary"):
    st.session_state.rodando, st.session_state.ult = not rodando, 0.0
    st.rerun()
intervalo = c[4].number_input("Intervalo (s)", 0.3, 10.0, 1.0, 0.1, key="intervalo")
if c[5].button("Cenário T1→T2→T3"):
    recarregar(True)
if c[6].button("Limpar"):
    recarregar(False)


# ---------- Visual do tanque ----------
def tanque_html(t):
    pct = t.fluido.volume / t.vol_max * 100
    over = " over" if t.overflow.volume > 0 else ""
    return f'<div class="tq{over}"><div class="lv" style="height:{min(100, pct):.1f}%"></div></div><div class="inf mut" style="text-align:center;width:56px">{pct:.0f}%</div>'


def info_html(t):
    f, th = t.fluido, (t.H["theta"][-1] if t.H["theta"] else None)
    sensor = t.H["sensor"][-1] if t.H["sensor"] else 0
    comp = "<br>".join(f"{e}: {m:.2f} g ({f.concentracao(e):.2f} g/L)" for e, m in f.massas.items() if m > 1e-9) or "vazio"
    s = f"<b>{f.nome}</b><br><b>{f.volume:.4f} L</b> · ρ {f.densidade:.4f} g/mL<br><span class='mut'>sensor {sensor:.4f}</span><br>{comp}"
    if th:
        s += "<br>" + " · ".join(f"{k.replace('theta ', 'θ')}={v:.1f}" for k, v in th.items())
        a6, a3 = th.get("theta 600"), th.get("theta 300")
        if a6 is not None and a3 is not None:
            s += f"<br>VA {a6/2:.1f} · VP {a6-a3:.1f} · LE {a3-(a6-a3):.1f}"
        if t.fora:
            s += "<br><span class='warn'>⚠ concentração fora da faixa de treino do RF</span>"
    elif f.volume > 0:
        s += "<br><span class='mut'>θ indisponível (sem modelo)</span>"
    if t.overflow.volume > 0:
        s += f"<br><span class='warn'>overflow: {t.overflow.volume:.3f} L</span>"
    return f'<div class="inf">{s}</div>'


def card(t):
    k = lambda s: f"{g}_{s}{t.id}"
    t.nome = st.text_input("Tanque", t.nome, key=k("n"))
    tl, tr = st.columns([1, 3])
    tl.markdown(tanque_html(t), unsafe_allow_html=True)
    tr.markdown(info_html(t), unsafe_allow_html=True)
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
    with st.expander("Configurações"):
        t.fluido.nome = st.text_input("Nome do fluido", t.fluido.nome, key=k("f"))
        t.vol_max = st.number_input("Volume máx (L)", 0.1, value=float(t.vol_max), step=0.5, key=k("v"))
        t.td = int(st.number_input("Sensor: atraso (passos)", 0, value=t.td, key=k("sd")))
        t.ruido = st.number_input("Sensor: ruído (%)", 0.0, value=t.ruido * 100, step=0.5, key=k("sr")) / 100
        if st.button("Remover tanque", key=k("rm")):
            bat.remover_tanque(t.id)
            st.rerun()


# ---------- Painel: atualiza sozinho enquanto estiver em Play ----------
def painel():
    ss = st.session_state
    if ss.rodando and time.time() - ss.ult >= 0.9 * intervalo:  # só avança no ritmo do timer
        bat.passo(1)
        ss.ult = time.time()
    st.caption(f"t = {bat.t}" + ("  ·  ▶ rodando" if ss.rodando else ""))

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

    st.subheader("Válvulas")
    for v in list(bat.valvulas.values()):
        c = st.columns([3, 3, 1.5, 1.2, 0.8])
        v.nome = c[0].text_input("Nome", v.nome, key=f"{g}_vn{v.id}", label_visibility="collapsed")
        c[1].markdown(f'<span class="dot{" on" if v.aberta else ""}"></span>{nm(v.origem)} → {nm(v.destino)}', unsafe_allow_html=True)
        v.vazao = c[2].number_input("L/passo", 0.0, value=float(v.vazao), step=0.05, key=f"{g}_vq{v.id}", label_visibility="collapsed")
        v.aberta = c[3].toggle("Aberta", v.aberta, key=f"{g}_va{v.id}")
        if c[4].button("✕", key=f"{g}_vx{v.id}"):
            del bat.valvulas[v.id]
            st.rerun()
    if bat.tanques:
        with st.expander("➕ Nova válvula"), st.form("nova_valvula", clear_on_submit=True):
            g1, g2, g3, g4 = st.columns(4)
            vn = g1.text_input("Nome da válvula")
            vo = g2.selectbox("Origem", list(bat.tanques), format_func=nm)
            vd = g3.selectbox("Destino", [None] + list(bat.tanques), format_func=nm)
            vq = g4.number_input("Vazão (L/passo)", 0.0, value=0.2, step=0.05)
            if st.form_submit_button("Criar") and vo != vd:
                bat.criar_valvula(vn or f"{nm(vo)} → {nm(vd)}", vo, vd, vq)
                st.rerun()

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


# run_every é definido na criação do fragmento: Play liga o timer, Pausar desliga
st.fragment(run_every=intervalo if rodando else None)(painel)()

with st.expander("Registro de operações"):
    st.text("\n".join(reversed(bat.log)))
st.download_button("⬇ Baixar CSV", bat.csv(), "simulacao.csv", "text/csv")
