import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
API_URL = "http://127.0.0.1:8000"   # if Streamlit runs on the same machine

# Configuration de la page
st.set_page_config(
    page_title='CarPriceML — Estimation du prix',
    layout='centered',
    page_icon='🚗'
)

st.title('🚗 CarPriceML — Estimation du prix (MAD)')
st.markdown(
    "### Entrez les caractéristiques de la voiture pour obtenir une estimation du prix en dirhams marocains (MAD)."
)

# Formulaire utilisateur
with st.form('predict_form'):
    col1, col2 = st.columns(2)
    with col1:
        company = st.text_input('Marque / Company', 'Maruti')
        model = st.text_input('Model', 'Swift')
        fuel = st.selectbox('Carburant', ['Diesel', 'Petrol', 'CNG', 'Electric'])
        seller_type = st.selectbox('Type de vendeur', ['Individual', 'Dealer'])
        transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    with col2:
        km_driven = st.number_input('Kilométrage (km)', value=50000, min_value=0)
        mileage_mpg = st.number_input('Mileage (mpg)', value=45.0, min_value=0.0)
        engine_cc = st.number_input('Engine (cc)', value=1200.0, min_value=500.0)
        max_power_bhp = st.number_input('Max Power (bhp)', value=80.0, min_value=10.0)
        seats = st.number_input('Seats', min_value=2, max_value=9, value=5, step=1)

    submitted = st.form_submit_button('🚀 Estimer le prix')

# Action après soumission
if submitted:
    payload = {
        'company': company,
        'model': model,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'km_driven': km_driven,
        'mileage_mpg': mileage_mpg,
        'engine_cc': engine_cc,
        'max_power_bhp': max_power_bhp,
        'seats': seats
    }

    with st.spinner('Estimation en cours...'):
        try:
            res = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            res.raise_for_status()
            data = res.json()
            predicted_price = data.get('predicted_price_mad', None)

            if predicted_price is not None:
                st.success(f"💰 Prix estimé : **{predicted_price:,.2f} MAD**")
            else:
                st.error("La réponse du serveur ne contient pas de prix estimé.")

        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de se connecter à l’API. Vérifie que le backend FastAPI est bien lancé.")
        except requests.exceptions.Timeout:
            st.error("⏳ La requête a expiré. Le serveur met trop de temps à répondre.")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Erreur lors de la requête : {str(e)}")
        except Exception as e:
            st.error(f"Une erreur inattendue est survenue : {str(e)}")

# Footer
st.markdown("---")
st.caption("© 2025 CarPriceML — Développé par Achraf Chair | MLOps & IA Project 🚀")
