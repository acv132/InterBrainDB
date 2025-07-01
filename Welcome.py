"""

This is the main page, that you have to run with "streamlit run" to launch the app locally.
Streamlit automatically create the tabs in the left sidebar from the .py files located in /pages
Here we just have the home page, with a short description of the tabs, and some images

"""
import os

import streamlit as st

# examples:
# https://neuromadlab.com/en/resources/living-meta-analysis-on-tvns-and-hrv/
# https://eitan177-cloneretriever-streamlit-cloneretriever-tt47bz.streamlit.app/
# https://hbretonniere-surviz--home-page-9r8djm.streamlit.app/Survey_footprint


st.set_page_config(
    page_title="Living Literature Review",
    page_icon="📚",
    layout="wide"
)

# Main content
st.title("📚 Living Literature Review")
st.markdown("""
### Welcome to the Living Literature Review

This tool is designed to visualize up-to-date research body related to remote hyperscanning by:

- **Displaying included studies** and their characteristics.
- **Allowing user-submitted studies** (with review).
- **Visualizing trends** and categorization.
- Providing **transparent inclusion/exclusion** overviews.

---
""")

st.markdown("""
### Purpose
This living literature review tracks emerging research on multimodal hyperscanning in collaborative contexts. The 
platform includes various categories, such as interaction scenario, measurement modalities, and analysis approaches, 
serving as a dynamic open-access resource. 
The aim is to provide a comprehensive overview of the current state of research, enabling researchers to stay informed about the latest developments in the field.

Want to contribute? Use the **Submit New Article** tab to add your studies, we regularly review submissions and add 
them to the database.

### Paper
Read all about the Living Literature Review in our paper:
```
[citation with doi]
```

### Code & Data
<a href="https://github.com/acv132/Hyperscanning-Living-Review" target="_blank">
    <img src="https://github.com/acv132/Hyperscanning-Living-Review/blob/main/assets/github-mark.png" height="40">
</a>

### Contact
Want to report an issue or suggest a feature? 
Write an e-mail to [anna.vorreuther@iat.universität-stuttgart.de](mailto:anna.vorreuther@iat.universität-stuttgart.de, "Subject: Living Literature Review Feedback")

""",
    unsafe_allow_html=True)

st.subheader("Funding")
image_extensions = [".png", ".jpg", ".jpeg"]
logo_folder = "./assets/logos_Funding"

for img in os.listdir(logo_folder):
    if any(img.endswith(ext) for ext in image_extensions):
        st.image(os.path.join(logo_folder, img), width=1000)
