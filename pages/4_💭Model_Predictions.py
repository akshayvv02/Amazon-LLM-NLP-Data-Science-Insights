import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import joblib
# import tensorflow as tf
from PIL import Image
import custom_functions as fn
import plotly.express as px
import plotly.io as pio
pio.templates.default='streamlit'
# Changing the Layout
st.set_page_config( layout="centered", 
                   page_icon='💭 ',
                   page_title="Model Predictions")


##Load in the data
import json
with open("config/filepaths.json") as f:
    FPATHS = json.load(f)

fpath_best_ml = FPATHS['results']['best-ml-clf_joblib']


# @st.cache_resource
def load_best_model_results(fpath_results_joblib):
    import joblib
    
    loaded = joblib.load(fpath_results_joblib)
    # if isinstance(loaded, dict):
    #     keys = list(loaded.keys())
    #     st.write(str(keys))
    # else:
    return loaded#joblib.load(fpath_results_joblib)

## Loading our training and test data
@st.cache_data
def load_Xy_data(joblib_fpath):
    return joblib.load(joblib_fpath)

    
## VISIBLE APP COMPONENTS START HERE
st.title("NLP Models & Predictions")
st.subheader("Predicting Amazon Review Rating")

# st.image("Images/dalle-yelp-banner-1.png",width=800,)
st.divider()

## VISIBLE APP COMPONENTS CONTINUE HERE
st.header("Get Model Predictions")

X_to_pred = st.text_area("### Enter text to predict here:", 
                         value="The noodles had a weird rubbery texture.")

## Lime Explanation Fucntions
from lime.lime_text import LimeTextExplainer
@st.cache_resource
def get_explainer(class_names = None):
	lime_explainer = LimeTextExplainer(class_names=class_names)
	return lime_explainer

def explain_instance(explainer, X_to_pred,predict_func):
	explanation = explainer.explain_instance(X_to_pred, predict_func, labels=(1,))
	return explanation.as_html(predict_proba=False)

# st.markdown("> Predict & Explain:")
get_any_preds = st.button("Get Predictions:")

get_pred_ml = True#st.checkbox("Machine Learning Model",value=True)
# get_pred_nn = st.checkbox("Neural Network", value=True)


def predict_decode(X_to_pred, best_ml_clf,lookup_dict):
    
    if isinstance(X_to_pred, str):
        X = [X_to_pred]
    else:
        X = X_to_pred

    # Get Predixtion
    pred_class = best_ml_clf.predict(X)[0]
    
    # In case the predicted class is missing from the lookup dict
    try:
        # Decode label
        class_name = lookup_dict[pred_class]
    except:
        class_name = pred_class
    return class_name


@st.cache_data
def load_target_lookup(encoder_fpath = FPATHS['metadata']['label_encoder_joblib']):
    # Load encoder and make lookup dict
    encoder = joblib.load(encoder_fpath)

    lookup_dict = {i:class_ for i,class_ in enumerate(encoder.classes_)}
    return encoder, lookup_dict


# Loading the ML model
@st.cache_resource
def load_ml_model(fpath):
    loaded_model = joblib.load(fpath)
    return loaded_model

encoder,target_lookup = load_target_lookup()

explainer = get_explainer(class_names=encoder.classes_)

best_ml_model = FPATHS['models']['ml']['logreg_joblib']
best_ml_clf = joblib.load(best_ml_model)
if isinstance(best_ml_clf, dict):
    # keys = list(best_ml_clf.keys())
    best_ml_clf= best_ml_clf['model']
    # st.write(str(keys))



# check_explain_preds  = st.checkbox("Explain predictions with Lime",value=False)
if (get_pred_ml) & (get_any_preds):
    st.markdown(f"> #### The ML Model predicted:")
    # with st.spinner("Getting Predictions..."):
    # st.write(f"[i] Input Text: '{X_to_pred}' ")
    pred = predict_decode(X_to_pred, lookup_dict=target_lookup,best_ml_clf=best_ml_clf)

    st.markdown(f"#### \t Rating=_{pred}_")
    st.markdown("> Explanation for how the words pushed the model towards its prediction:")
    explanation_ml = explain_instance(explainer, X_to_pred, best_ml_clf.predict_proba )#lime_explainer.explain_instance(X_to_pred, best_ml_clf.predict_proba,labels=label_index_ml)
    with st.container():
        components.html(explanation_ml,height=800)
else: 
    st.empty()


st.divider()

st.header("Model Evaluation")


# st.subheader("Machine Learning Model")
# c1, c2 = st.columns(2)
with st.spinner("Loading model results..."):
    results = load_best_model_results(FPATHS['results']['best-ml-clf_joblib'])
        
        
# st.subheader("Model Evaluation Results")
with st.expander("Model Parameters:"):
    st.write(results['model'].get_params())

with st.expander("Show results for the test data.",expanded=True):
    st.text(results['test']['classification_report'])
    st.pyplot(results['test']['confusion_matrix'])
    st.text("\n\n")

with st.expander("Show results for the training data."):
    st.text(results['train']['classification_report'])
    st.pyplot(results['train']['confusion_matrix'])
    st.text("\n\n")


st.divider()

## Author Information
with open("app-assets/author-info.md") as f:
    author_info = f.read()
    
with st.sidebar.container(border=True):
    st.subheader("Author Information")
    st.markdown(author_info, unsafe_allow_html=True)
    