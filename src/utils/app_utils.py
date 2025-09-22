from __future__ import annotations

import base64
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
import streamlit as st
from bs4 import BeautifulSoup
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles
from htbuilder.funcs import rgba, rgb
from htbuilder.units import percent, px


def set_mypage_config():
    st.set_page_config(
        page_title="InterBrainDB - Living Literature Review", page_icon='assets/favicon.ico', layout="wide"
        )
    ms = st.session_state
    if "themes" not in ms:
        ms.themes = {
            "current_theme": "light", "refreshed": True,

            "dark": {
                "theme.base": "dark",
                "theme.backgroundColor": "#1e1f22",
                "theme.primaryColor": "#019879",
                "theme.secondaryBackgroundColor": "#2b2d30",
                "theme.textColor": "#FAFAFA",
                "theme.font": "sans serif",
                "button_face": ":material/dark_mode:"
                },

            "light": {
                "theme.base": "light",
                "theme.backgroundColor": "#FFFFFF",
                "theme.primaryColor": "#ADE1DC",
                "theme.secondaryBackgroundColor": "#F0F2F6",
                "theme.textColor": "#1e1f22",
                "theme.font": "sans serif",
                "button_face": ":material/light_mode:"
                },
            }

    def ChangeTheme():
        previous_theme = ms.themes["current_theme"]
        st.write(previous_theme)
        tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
        for vkey, vval in tdict.items():
            if vkey.startswith("theme"): st._config.set_option(vkey, vval)

        ms.themes["refreshed"] = False
        if previous_theme == "dark":
            ms.themes["current_theme"] = "light"
        elif previous_theme == "light":
            ms.themes["current_theme"] = "dark"

    btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"][
        "button_face"]
    col1, col2 = st.columns([10, 1])
    with col2:
        st.button(btn_face, on_click=ChangeTheme)

    if ms.themes["refreshed"] == False:
        ms.themes["refreshed"] = True
        st.rerun()


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 80px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color=st.get_option("theme.textColor"),
        text_align="center",
        height="auto",
        opacity=1
        )

    style_hr = styles(
        display="block", margin=px(8, 8, "auto", "auto"), border_style="inset", border_width=px(0)
        )

    body = p()
    foot = div(
        style=style_div
        )(
        hr(
            style=style_hr
            ), body
        )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def _svg_data_uri(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/svg+xml;base64,{b64}"


def footer():
    svg_uri = _svg_data_uri("assets/streamlit.svg")
    myargs = ["Made in ", link(
        "https://streamlit.io/", image(svg_uri, width=px(25), height=px(25))
        ), br(), link(
        "https://dsi-generator.fraunhofer.de/impressum/impressum_view/en/ff3d5595-4141-4548-9b79-40cb3bb71a91/",
        "Imprint"
        ), " | ", link(
        "https://dsi-generator.fraunhofer.de/dsi/view/en/ff12fb5e-3c20-4155-b7e6-7fdaf82ee9d5/",
        "Privacy Policy"
        ), br(), ]
    layout(*myargs)


@st.cache_data(ttl=0)  # cache for 1 hour
def _fetch_imprint_html(url: str) -> str:
    # Fetch with a friendly UA; some servers block default python user agents
    headers = {"User-Agent": "Mozilla/5.0 (Streamlit-Impressum/1.0)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    # return r.text
    html = r.text

    # CSS style to enforce font
    color = st.get_option("theme.textColor")
    style_block = (
        '<style>body { '
        'font-family: "Source Sans Pro", sans-serif; '
        'line-height: 1.6;'
        ' color: ' + color +
        '}</style>'
    )

    # Try to insert inside <head>, otherwise prepend
    if re.search(r"<head.*?>", html, flags=re.IGNORECASE | re.DOTALL):
        html = re.sub(
            r"(<head.*?>)",
            r"\1\n" + style_block,
            html,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )
    else:
        html = style_block + html
    return html


def _absolutize_links(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Fix <a>, <img>, <link>, <script> URLs to be absolute
    for tag, attr in (("a", "href"), ("img", "src"), ("link", "href"), ("script", "src")):
        for el in soup.find_all(tag):
            if el.has_attr(attr):
                el[attr] = urljoin(base_url, el[attr])

    # Optional: strip any inline scripts for safety in st.markdown
    for s in soup.find_all("script"):
        s.decompose()

    # Optional: basic style normalization to blend with Streamlit theme
    style = soup.new_tag("style")
    style.string = """
      body { margin: 0; padding: 0; }
      .impressum, .imprint, main { color: inherit; }
      a { text-decoration: none; }
      a:hover { text-decoration: underline; }
    """
    soup.head.append(style) if soup.head else soup.insert(0, style)
    return str(soup)


def render_fraunhofer_impressum(url: str):
    st.markdown("### 🧾 Imprint")
    st.caption("Automatically embedded.")

    try:
        html = _fetch_imprint_html(url)
        html = _absolutize_links(html, url)
        # Render inside an isolated iframe so layout/styles don’t clash
        st.components.v1.html(html, height=900, scrolling=True)
        with st.expander("Direct link"):
            st.write(url)
    except Exception as e:
        st.warning("Error retrieving the legal notice. You may find the direct link below.")
        st.write(f"[Directly open imprint]({url})")
        st.caption(f"Technical details: {e}")


def render_fraunhofer_privacy_policy(url: str):
    st.markdown("### 🧾 Privacy Policy")
    st.caption("Automatically embedded.")

    try:
        html = _fetch_imprint_html(url)
        html = _absolutize_links(html, url)
        # Render inside an isolated iframe so layout/styles don’t clash
        st.components.v1.html(html, height=900, scrolling=True)
        with st.expander("Direct link"):
            st.write(url)
    except Exception as e:
        st.warning("Error retrieving the legal notice. You may find the direct link below.")
        st.write(f"[Directly open privacy policy]({url})")
        st.caption(f"Technical details: {e}")


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
