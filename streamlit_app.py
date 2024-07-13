import streamlit as st
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

df = pd.read_csv('creditcard.csv')

def main():
    
    nav = st.sidebar.radio('Navigation',['Analytics','Prediction'])
    if st.sidebar.button('Exit'):
        st.title('Credit Card Approval')
        st.empty()
        st.title('Thank you!')
        st.stop()
    
    if nav == 'Analytics':
        st.title('Analytics of Credit Card Approval Trends')

        st.header('Heatmap')
        figure = plt.figure(figsize=[10,10])
        sns.heatmap(df.corr(),annot=True,cmap='crest')
        st.pyplot(figure)

        st.header('Pairplot')
        fig= sns.pairplot(df)
        st.pyplot(fig)

        st.header('Bar Plot')
        st.bar_chart(data=df, x='card',y='age')
        st.write('Income vs Card Approval')
        st.bar_chart(data=df, x='card',y='income')
        st.write('Ownership of House vs Card Approval')
        st.bar_chart(data=df, x='card',y='owner')
        st.write('Self Employment vs Card Approval')
        st.bar_chart(data=df, x='card',y='selfemp')
        st.write('Number of months of stay at given location vs Card Approval')
        st.bar_chart(data=df, x='card',y='months')
        st.write('Major Cards owned vs Card Approval')
        st.bar_chart(data=df, x='card',y='majorcards')

        st.header('Pie Chart')
        st.write('Distribution of number of reports')
        grouped_report = df['reports'].value_counts().rename_axis('value').reset_index(name='counts')
        reports_pie = px.pie(grouped_report, values='counts', names='value')
        st.plotly_chart(reports_pie)
        st.write('Distribution of number of dependents')
        grouped_dependents = df['dependents'].value_counts().rename_axis('value').reset_index(name='counts')
        reports_dep = px.pie(grouped_dependents, values='counts', names='value')
        st.plotly_chart(reports_dep)
        st.write('Distribution of number of active cards')
        grouped_act = df['active'].value_counts().rename_axis('value').reset_index(name='counts')
        reports_act = px.pie(grouped_act, values='counts', names='value')
        st.plotly_chart(reports_act)


    if nav == 'Prediction':
        # Load the trained model
        model = pickle.load(open("rforest.pkl", "rb"))

        # Create a Streamlit app
        st.title("Predictor")

        # Create input fields for user input
        st.header("Enter your information:")
        reports = st.number_input('Number of major derogatory reports')
        age = st.text_input('Age', 30)
        income = st.number_input("Yearly Income")
        own = st.radio('Own house?', ['Yes', 'No'])
        semp = st.radio('Self employed?', ['Yes', 'No'])
        dependents = st.number_input("Number of people in family(including yourself)")
        months = st.number_input("Number of months living at current location: ")
        majorcards = st.number_input("Number of major credit cards held: ")
        active = st.number_input("Number of active credit accounts: ")

        #change the string input to numeric
        if own == "Yes":
            owner = 1
        else:
            owner = 0

        if semp == "Yes":
            selfemp = 1
        else:
            selfemp = 0

        #create submit button
        submit_button = st.button("Submit")

        #predict when submit button clicked
        if submit_button:
            prediction = model.predict([[reports,age,income,owner,selfemp,dependents,months,majorcards,active]])
            st.header("Prediction:")
            if prediction == 1:
                st.write("You are likely to be approved for a credit card!")
            else:
                st.write("You are unlikely to be approved for a credit card.")

if __name__ == '__main__':
    main()