# IPL 2027 Prediction

Streamlit deployment of the supplied educational IPL project.

The supplied model uses fixed encoded demo inputs. Training data and categorical encoders are missing, so the app does not yet predict from selected teams. See ORIGINAL_README.md and ipl.ipynb for original source attribution.

## Run
Use Python 3.11, install requirements.txt, and run `streamlit run app.py`.

## Render
Free web service in Singapore. Build: `pip install -r requirements.txt`. Start: `streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true`. Health endpoint: `/_stcore/health`. No database or API key needed.
