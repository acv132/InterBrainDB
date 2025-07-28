"""

This is the main page, that you have to run with "streamlit run" to launch the app locally.
Streamlit automatically create the tabs in the left sidebar from the .py files located in /pages
Here we just have the home page, with a short description of the tabs, and some images

"""

import base64

import streamlit as st

# examples:
# https://neuromadlab.com/en/resources/living-meta-analysis-on-tvns-and-hrv/
# https://eitan177-cloneretriever-streamlit-cloneretriever-tt47bz.streamlit.app/
# https://hbretonniere-surviz--home-page-9r8djm.streamlit.app/Survey_footprint


# ========================
# 💅 UI Configuration
# ========================
st.set_page_config(
    page_title="Living Literature Review",
    page_icon='assets/favicon.ico',
    layout="wide"
)
st.title("📚 Living Literature Review on Digital Hyperscanning")
st.subheader("Welcome to the Living Literature Review")
st.markdown("""
This tool is designed to visualize up-to-date research body related to hyperscanning with digital components by:

- **Displaying included studies** and their characteristics.
- **Visualizing trends** and (subjective) categorization.
- Integrating **user-submitted studies**.
- Providing **transparent inclusion/exclusion** overviews related to the original paper.

---
""")
st.subheader("Purpose")
st.markdown("""
This living literature review tracks emerging research on multimodal hyperscanning in contexts with a digital 
component. The platform includes various categories, such as interaction scenario, type of tasks, measurement 
modalities, and analysis approaches, serving as a dynamic open-access resource. 
The aim is to provide a comprehensive overview of the current state of digital hyperscanning research, 
enabling researchers to stay informed about the latest developments in the field.
Eventually, the platform will also include hyperscanning studies that do not involve a digital component, as well as 
extending the scope to include other population groups, such as children or clinical populations.
""")
st.subheader("Adding New Articles")
st.markdown("A living review means a growing database. Therefore, if we either missed a paper or you published "
            "results, please consider submitting your article. Click the button below to submit a new article for "
            "review:")
st.page_link(label="Submit New Article", page="pages/2_Submit_New_Article.py", icon="🆕")

st.subheader("Paper")
st.markdown(f"""
Read the original literature review here:
[Link Button]
```
[citation with doi]
```
If you use this resource, please consider citing our paper. 
""")

st.subheader("Code")
st.markdown("Want to report an issue or suggest a feature? Post an issue on the GitHub repository.")
label = "Git Repository"
url = "https://github.com/acv132/Hyperscanning-Living-Review"
icon_path = "assets/github.svg"
icon_width =35  # in pixels
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

st.markdown("""
### Contact
The app is being maintained by the Applied Neurocognitive Systems Team, Department of Human-Computer Interaction, 
University of Stuttgart & Fraunhofer IAO.

Contact: Anna Vorreuther [anna.vorreuther@iat.universität-stuttgart.de](mailto:anna.vorreuther@iat.universität-stuttgart.de,
"Subject: Living Literature Review Contact Request")
""")



st.subheader("Funding and Support")
logo = "./assets/logos.svg"
st.image(logo, width=1000)
