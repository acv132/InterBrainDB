"""
This is the main page, that you have to run with "streamlit run" to launch the app locally.
Streamlit automatically create the tabs in the left sidebar from the .py files located in /pages
Here we just have the Welcome page, with a short description of the tabs, and some images
"""
# ========================
# 📦 Imports & Setup
# ========================
from __future__ import annotations
import base64
import streamlit as st

from src.utils.app_utils import footer, clickable_image, set_mypage_config

# ========================
# 💅 UI Configuration
# ========================
set_mypage_config()
st.title("📚 InterBrainDB")
st.subheader("Welcome to the Living Literature Review Database on Hyperscanning")
st.markdown(
    """
    This tool is designed to visualize up-to-date research body related to hyperscanning. The following features are 
    included:
    
    - **Displaying studies:** An overview of included studies can be found in table format and descriptive 
    information is visualized, including filters for specific searches.
    - **Categorization:** Studies are categorized based on population, design, analyses, and outcomes to derive 
    clusters and shared traits.
    - **Submission system:** Further studies may be submitted by the community.
    - **Replication of original review:** An overview of studies included in the original review paper is provided to 
    replicate reported findings and extend analyses with updated versions of the database.
    
    ---
    """
    )

st.subheader("Purpose")
st.markdown(
    """
    This living literature review tracks emerging research on multimodal hyperscanning in contexts with a digital 
    component. The platform includes various categories, such as interaction scenario, type of tasks, measurement 
    modalities, and analysis approaches, serving as a dynamic open-access resource. 
    The aim is to provide a comprehensive overview of the current state of digital hyperscanning research, 
    enabling researchers to stay informed about the latest developments in the field.
    Eventually, the platform will also include additional tags, e.g., “open-source dataset available” and the 
    scope will be extended to include further categories, such as studies with under-aged or clinical populations.
    """
    )

st.subheader("Adding New Articles")
st.markdown(
    "A living review means a growing database. Therefore, if we either missed a paper or you published "
    "new results, please consider submitting your article. Click the button below to submit a new article for "
    "review:"
    )
st.page_link(label="Submit New Article", page="src/pages/3_Submit_New_Article.py", icon="🆕")

st.subheader("Paper")
st.markdown(
    f"""
Read the original literature review here:
[Link Button](tbd)
```
[citation with doi]
```
If you use this resource, please consider citing our paper. 
"""
    )

st.subheader("Code")
st.markdown("Want to report an issue or suggest a feature? Post an issue on the GitHub repository.")
label = "Git Repository"
url = "https://github.com/acv132/InterBrainDB"
icon_path = "assets/github.svg" if st.get_option("theme.base") == "light" else "assets/github_dark.svg"
icon_width = 35  # in pixels
with open(icon_path, "rb") as f:
    svg_bytes = f.read()
b64 = base64.b64encode(svg_bytes).decode()
html = f"""
<a href="{url}" target="_blank"
   style="text-decoration: none; display: inline-flex; align-items: center;">
  <img src="data:image/svg+xml;base64,{b64}" width="{icon_width}" style="margin-right:8px;"/>
  <span style="font-size:16px; color:inherit;">{label}</span>
</a>
"""
st.markdown(html, unsafe_allow_html=True)
st.markdown(
    """
    ### Contact
    The app is maintained by the Applied Neurocognitive Systems Team, Department of Human-Computer Interaction, 
    University of Stuttgart & Fraunhofer IAO.
    
    Contact: Anna Vorreuther [anna.vorreuther@iat.uni-stuttgart.de](mailto:anna.vorreuther@iat.uni-stuttgart.de,
    "Subject: InterBrainDB Contact Request")
    """
    )

st.subheader("Funding and Support")
SPONSORS = [{
    "name": "Applied Neurocognitive Systems",
    "path": "./assets/logos/ANS_dark.svg" if st.get_option("theme.base") == "dark" else "./assets/logos/ANS.svg",
    "url": "https://linktr.ee/ans_iao",
    "alt": "Applied Neurocognitive Systems",
    }, {
    "name": "Institut für Arbeitswissenschaft und Technologiemanagement (IAT)",
    "path": "./assets/logos/IAT_de_dark.svg" if st.get_option("theme.base") == "dark" else "./assets/logos/IAT_de.svg",
    "url": "https://www.iat.uni-stuttgart.de/",
    "alt": "Institut für Arbeitswissenschaft und Technologiemanagement (IAT)",
    }, {
    "name": "Fraunhofer IAO",
    "path": "./assets/logos/FraunhoferIAO_dark.svg" if st.get_option(
        "theme.base"
        ) == "dark" else "./assets/logos/FraunhoferIAO.svg",
    "url": "https://www.iao.fraunhofer.de/",
    "alt": "Fraunhofer IAO",
    }, {
    "name": "Radboud University",
    "path": "./assets/logos/radboud_dark.svg" if st.get_option(
        "theme.base"
        ) == "dark" else "./assets/logos/radboud.svg",
    "url": "https://www.ru.nl/en",
    "alt": "Radboud University",
    }, {
    "name": "TNO",
    "path": "./assets/logos/tno_dark.svg" if st.get_option("theme.base") == "dark" else "./assets/logos/tno.svg",
    "url": "https://www.tno.nl/",
    "alt": "TNO",
    }, ]

cols_per_row = 5  # tweak to taste
rows = [SPONSORS[i:i + cols_per_row] for i in range(0, len(SPONSORS), cols_per_row)]

for row in rows:
    cols = st.columns(len(row), gap="large")
    for col, sponsor in zip(cols, row):
        with col:
            clickable_image(sponsor["path"], sponsor["url"], alt=sponsor["alt"], width=500)
st.markdown(
    """
    <style>
      .stMarkdown a img { transition: transform .1s ease-in-out; }
      .stMarkdown a:hover img { transform: scale(1.03); }
    </style>
    """, unsafe_allow_html=True, )

footer()
