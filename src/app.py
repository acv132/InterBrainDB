import streamlit as st

pg = st.navigation([st.Page("pages/1_Welcome.py"),
                    st.Page("pages/2_Data.py"),
                    st.Page("pages/3_Submit_New_Article.py"),
                    st.Page("pages/4_Imprint_&_Privacy_Policy.py"),
                    ])
pg.run()
