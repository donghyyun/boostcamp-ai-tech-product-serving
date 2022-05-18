import pandas as pd
import streamlit as st

from confirm_button_hack import cache_on_button_press
from model import CatBoostModel

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")
root_password = 'donghyyun'
DATA_URL = "./processed_test_data_for_streamlit.csv"
SHOW_FEAT = ["user", "assessmentItemID"]

def main():
    st.title("Special mission 1: front-end with streamlit")
    data = st.cache(pd.read_csv)(DATA_URL)
    model = CatBoostModel()
    
    # Select some rows using st.multiselect.
    st.write('### Test Dataset Candidates', data[SHOW_FEAT])
    st.write("## ")
    st.subheader('Please select rows you want to test:')
    selected_indices = st.multiselect('', data.index)
    
    st.write('### Selected Rows', data[SHOW_FEAT].values[selected_indices])
    
    if st.button("inference"):
        st.subheader('Inference Result(The probability that the user will get the correct answer)')
        st.write(model.inference(data.iloc[selected_indices]))
    

@cache_on_button_press('Authenticate')
def authenticate(password) -> bool:
    return password == root_password


password = st.text_input("type your password", type="password")

if authenticate(password):
    st.success("Authorized!!")
    main()
else:
    st.error("Invalid password")
