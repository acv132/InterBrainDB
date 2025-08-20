from urllib.parse import urljoin

import requests
import streamlit as st
from bs4 import BeautifulSoup
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles
from htbuilder.funcs import rgba, rgb
from htbuilder.units import percent, px


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
        color="black",
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


def footer():
    myargs = ["Made in ", link(
        "https://streamlit.io/", image(
            'https://avatars3.githubusercontent.com/u/45109972?s=400&v=4', width=px(25), height=px(25)
            )
        ), br(), link(
        "https://dsi-generator.fraunhofer.de/impressum/impressum_view/en/ff3d5595-4141-4548-9b79-40cb3bb71a91/",
        "Imprint"
        ), " | ", link(
        "https://dsi-generator.fraunhofer.de/dsi/view/en/ff12fb5e-3c20-4155-b7e6-7fdaf82ee9d5/", "Privacy Policy"
        ), br(), ]
    layout(*myargs)


@st.cache_data(ttl=60 * 60)  # cache for 1 hour
def _fetch_imprint_html(url: str) -> str:
    # Fetch with a friendly UA; some servers block default python user agents
    headers = {"User-Agent": "Mozilla/5.0 (Streamlit-Impressum/1.0)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.text


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
