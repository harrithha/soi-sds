import streamlit as st


class MultiApp:
    def __init__(self):
        self.apps = []
        self.titles = []

    def num_apps(self):
        return len(self.apps)

    def add_app(self, title, func):
        self.apps.append(func)
        self.titles.append(title)

    def get_index(self, title):
        return self.titles.index(title)

    def run(self):
        app = st.radio('',
                       self.titles)
        st.markdown(""" <style>
            div[role="radiogroup"]
            {
                position: absolute !important;
                bottom: 40px !important;
                left: -380px !important;
            }
            footer {visibility: hidden;}</style> """, unsafe_allow_html=True)
        self.apps[self.get_index(app)]()
