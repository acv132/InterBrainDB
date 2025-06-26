"""

This is the main page, that you have to run with "streamlit run" to launch the app locally.
Streamlit automatically create the tabs in the left sidebar from the .py files located in /pages
Here we just have the home page, with a short description of the tabs, and some images

"""
import os

import streamlit as st

# todo implement frontend
#  sidebar navigation: filters
#       Include/exclude specific studies
#       publication year
#       per categories
#       info hover buttons for each filter
#       reload button
#  tab navigation (top): pages
#  page: home page
#       welcome
#       Purpose
#       explanation
#       paper ref
#       code + data ref
#       adding new articles (submit page)
#       contact
#       funding
#       links
#  page: data overview (table of currently included studies with filters applied)
#  page: data display
#   display data tables
#   display interaction figure
#   diplay counts per category
#   combine up to three categories to create a 3D cluster plot of studies
#  page: submit new data point (article) with mandatory and optional fields to fill in
#  optional: paper review overview (included and excluded full-text screening overview; exclusion reason PICOS)

# examples:
# https://neuromadlab.com/en/resources/living-meta-analysis-on-tvns-and-hrv/
# https://eitan177-cloneretriever-streamlit-cloneretriever-tt47bz.streamlit.app/
# https://hbretonniere-surviz--home-page-9r8djm.streamlit.app/Survey_footprint

# todo implement backend
#  load in database (excel file)
#       how to treat missing values (can there be any?)
#  update "used data" var after filters are set (reload button logic)
#  create figures
#  host page on server
#  store submit requests and send notification via e-mail

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
[Short explanation of the meta-analysis goal]
### Paper
[citation with copy function and doi]
### Code & Data
[GitHub]
""")

st.subheader("Funding")
image_extensions = [".png", ".jpg", ".jpeg"]
logo_folder = "./assets/logos"
st.markdown("todo change svg logos to valid extensions")

for img in os.listdir(logo_folder):
    if any(img.endswith(ext) for ext in image_extensions):
        st.image(os.path.join(logo_folder, img), width=150)
