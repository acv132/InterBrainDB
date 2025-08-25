"""

This is the main page, that you have to run with "streamlit run" to launch the app locally.
Streamlit automatically create the tabs in the left sidebar from the .py files located in /pages
Here we just have the Welcome page, with a short description of the tabs, and some images

"""

# todo add impressum and data privacy aspects for web hosting

from __future__ import annotations
import streamlit as st
import base64
from pathlib import Path
import streamlit as st

from plotting.plot_utils import is_dark_color
from utils.app_utils import footer

# ========================
# 💅 UI Configuration
# ========================
st.set_page_config(
    page_title="Living Literature Review", page_icon='assets/favicon.ico', layout="wide"
    )
st.title("📚 Living Literature Review on Digital Hyperscanning")
st.subheader("Welcome to the Living Literature Review")
st.markdown(
    """
    This tool is designed to visualize up-to-date research body related to hyperscanning with digital components by:
    
    - **Displaying included studies** and their characteristics.
    - **Visualizing trends** and (subjective) categorization.
    - Integrating **user-submitted studies**.
    - Providing **transparent inclusion/exclusion** overviews related to the original paper.
    
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
    Eventually, the platform will also include hyperscanning studies that do not involve a digital component, as well as 
    extending the scope to include other population groups, such as children or clinical populations.
    """
    )
st.subheader("Adding New Articles")
st.markdown(
    "A living review means a growing database. Therefore, if we either missed a paper or you published "
    "results, please consider submitting your article. Click the button below to submit a new article for "
    "review:"
    )
st.page_link(label="Submit New Article", page="pages/2_Submit_New_Article.py", icon="🆕")

st.subheader("Paper")
st.markdown(
    f"""
Read the original literature review here:
[Link Button]
```
[citation with doi]
```
If you use this resource, please consider citing our paper. 
"""
    )

st.subheader("Code")
st.markdown("Want to report an issue or suggest a feature? Post an issue on the GitHub repository.")
label = "Git Repository"
url = "https://github.com/acv132/Hyperscanning-Living-Review"
icon_path = "assets/github.svg"
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
    
    Contact: Anna Vorreuther [anna.vorreuther@iat.universität-stuttgart.de](mailto:anna.vorreuther@iat.universität-stuttgart.de,
    "Subject: Living Literature Review Contact Request")
    """
    )

st.subheader("Funding and Support")
logo = "./assets/logos.svg"
st.image(logo, width=1000)
# tno-ifl.svg
# Logo-TNO.svg
# FraunhoferIAO.svg
# radboud.svg
# IAT_de.svg
# todo add logos separately with links to websites embedded in the images


# ---------- Helpers ----------
def _b64(path: Path) -> str:
    """Read a file and return base64-encoded string."""
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")

def clickable_image(path: str | Path, href: str, *, alt: str = "", width: int | None = 160):
    """
    Render a clickable image (SVG or raster) that opens in a new tab.
    For SVG we embed as data URI to ensure reliable rendering.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".svg":
        src = f"data:image/svg+xml;base64,{_b64(p)}"
    elif ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        mime = "png" if ext == ".png" else "jpeg" if ext in {".jpg", ".jpeg"} else ext.strip(".")
        src = f"data:image/{mime};base64,{_b64(p)}"
    else:
        # If unknown type, let Streamlit try rendering it directly
        with st.container():
            st.link_button(f"Open {alt or p.name}", href, use_container_width=True)
            st.image(str(p), width=width)
        return

    style_w = f"width:{width}px;" if width else "max-width:100%;"
    html = f"""
    <a href="{href}" target="_blank" rel="noopener noreferrer" title="{alt}">
      <img src="{src}" alt="{alt}" style="{style_w} display:block; margin:auto;" />
    </a>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---------- Data (paths + links) ----------
# Update any URLs if your partners differ.
SPONSORS = [{
    "name": "Applied Neurocognitive Systems", "path": "./assets/ANS.svg", "url": "https://linktr.ee/ans_iao",
    "alt": "Applied Neurocognitive Systems",
    }, {
    "name": "Institut für Arbeitswissenschaft und Technologiemanagement (IAT)",
    "path": "./assets/IAT_de.svg",
    "url": "https://www.iat.uni-stuttgart.de/",
    "alt": "Institut für Arbeitswissenschaft und Technologiemanagement (IAT)",
    },
    {
        "name": "Fraunhofer IAO",
        "path": "./assets/FraunhoferIAO.svg",
        "url": "https://www.iao.fraunhofer.de/",
        "alt": "Fraunhofer IAO",
    },
    {
        "name": "Radboud University",
        "path": "./assets/radboud.svg",
        "url": "https://www.ru.nl/en",
        "alt": "Radboud University",
    },
    {
        "name": "TNO",
        "path": "./assets/tno-ifl.svg" if is_dark_color(st.get_option('theme.backgroundColor')) else "./assets/Logo-TNO.svg",
        "url": "https://www.tno.nl/",
        "alt": "TNO",
    },
]

# ---------- Grid layout ----------
cols_per_row = 5  # tweak to taste
rows = [SPONSORS[i:i+cols_per_row] for i in range(0, len(SPONSORS), cols_per_row)]

for row in rows:
    cols = st.columns(len(row), gap="large")
    for col, sponsor in zip(cols, row):
        with col:
            clickable_image(sponsor["path"], sponsor["url"], alt=sponsor["alt"], width=500)
            st.caption(f"[{sponsor['name']}]({sponsor['url']})")

# ---------- Optional: subtle hover style for images ----------
st.markdown(
    """
    <style>
      .stMarkdown a img { transition: transform .1s ease-in-out; }
      .stMarkdown a:hover img { transform: scale(1.03); }
    </style>
    """,
    unsafe_allow_html=True,
)

footer()