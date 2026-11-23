import json

import pandas as pd
import streamlit as st

from engine.config import RESULTS_DIR

st.set_page_config(page_title="FinPulse: Model Evaluation", layout="wide")
st.title("Model evaluation: FinBERT vs. baselines")

path = RESULTS_DIR / "model_eval.json"
if not path.exists():
    st.info("No results yet. Run `pip install -r requirements-dev.txt` and then "
            "`python scripts/evaluate_models.py`.")
    st.stop()

data = json.loads(path.read_text())
st.markdown(f"Dataset: **{data['dataset']}**. Generated {data['generated_at']}.")
st.warning(data["caveat"])

df = pd.DataFrame(data["results"])
st.dataframe(df, hide_index=True, width="stretch",
             column_config={k: st.column_config.NumberColumn(format="%.3f")
                            for k in ["accuracy", "macro_f1", "weighted_f1"]})
for subset, g in df.groupby("subset", sort=False):
    st.subheader(f"Subset: {subset}")
    st.bar_chart(g.set_index("model")[["accuracy", "macro_f1"]], stack=False, height=280)
