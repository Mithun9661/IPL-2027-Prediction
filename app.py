from pathlib import Path
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IPL 2027 Prediction", page_icon="🏏")
st.title("🏏 IPL 2027 Prediction")
st.warning("Educational demo: training data and categorical encoders are missing. Real team-based prediction is not available yet.")
st.write("The supplied model can run on fixed encoded sample inputs only. Its numeric output cannot reliably be mapped to a team without the original encoders.")

@st.cache_resource
def load_model():
    with Path(__file__).with_name("ipl_model.pkl").open("rb") as source:
        return pickle.load(source)

sample = pd.DataFrame([[1, 2, 1, 0, 5]], columns=["team1", "team2", "toss_winner", "toss_decision", "venue"])
st.caption("Original encoded demo input")
st.dataframe(sample, hide_index=True)
if st.button("Run model demo"):
    model = load_model()
    prediction = model.predict(sample)
    st.info(f"Encoded output class: {prediction[0]}")
    st.caption("This is a model execution demo, not a forecast for selected teams or the IPL 2027 tournament.")
