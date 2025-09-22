import streamlit as st

pg = st.navigation([st.Page("src/pages/1_Welcome.py"),
                    st.Page("src/pages/2_Database.py"),
                    st.Page("src/pages/3_Submit_New_Article.py"),
                    st.Page("src/pages/4_Imprint_&_Privacy_Policy.py"),
                    ])
pg.run()
