import streamlit as st

st.write("WELCOME")
PASSWORD = "laws70217"

def check_password():
    if st.session_state.get("authenticated"):
        return True
    pwd = st.text_input("Password", type="password")
    if pwd:
        if pwd == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False

if not check_password():
    st.stop()
def check_password():
    if st.session_state.get("authenticated"):
        return True

    pwd = st.text_input("Password", type="password")
    if pwd:
        if pwd == st.secrets.get("APP_PASSWORD"):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False

if not check_password():
    st.stop()

import streamlit as st

st.write("WELCOME")