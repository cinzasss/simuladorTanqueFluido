# Simulador de tanques (Streamlit)

Fluido modular (`Fluido`), tanques, válvulas e previsão de θ600/θ300/θ200/θ100 por Random Forest.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Dados
- Local: coloque `dados/Dados experimentais.xlsx` (aba `Resultados`; colunas `CMC`, `Barita`, `Sal`, `theta 600`, `theta 300`, `theta 200`, `theta 100`). Outro caminho: variáveis `DADOS_XLSX` e `DADOS_ABA`.
- Streamlit Cloud: ou versione o Excel no repositório, ou envie-o pelo uploader da barra lateral (não precisa estar no GitHub). Sem Excel o app roda, só sem θ.

## Estrutura
- `simulador.py`: núcleo sem interface (`Fluido`, `PredVisc`, `Tanque`, `Valvula`, `Bateria`).
- `streamlit_app.py`: interface.

## Notas
- O RF recebe concentração em g/L; confira a unidade do Excel.
- Fora da faixa de treino o RF não extrapola; o app mostra um aviso.
- Cada dosagem/transferência avança 1 passo; "Rodar n passos" avança n com as válvulas abertas escoando.
- O estado fica em `st.session_state` (um por sessão do navegador).
