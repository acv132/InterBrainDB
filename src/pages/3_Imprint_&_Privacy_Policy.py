# ========================
# 📦 Imports & Setup
# ========================
import streamlit as st
from utils.app_utils import render_fraunhofer_impressum, render_fraunhofer_privacy_policy, set_mypage_config

# ========================
# 💅 UI Configuration
# ========================
set_mypage_config()

st.title("Imprint / Impressum & Privacy Policy")
if __name__ == "__main__":
    render_fraunhofer_impressum(
        "https://dsi-generator.fraunhofer.de/impressum/impressum_view/en/ff3d5595-4141-4548-9b79-40cb3bb71a91/"
        )
    st.divider()
    render_fraunhofer_privacy_policy(
        "https://dsi-generator.fraunhofer.de/dsi/view/en/ff12fb5e-3c20-4155-b7e6-7fdaf82ee9d5/"
        )
