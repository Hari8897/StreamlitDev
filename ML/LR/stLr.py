import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Use credentials to access the database
if 'users' not in st.session_state:
    st.session_state.users = {
        "admin": "admin123",
        "user1": "pass123",
        "user2": "abc456"
    }
    
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    
#2 login and register page
if not st.session_state.logged_in:
    st.title("🔐 Login Page")
    menu = st.selectbox("Select Menu", ["Login", "Create Account"])

    if menu == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.success(f"Welcome, {username}!")
                st.session_state.logged_in = True
               
            else:
                st.error("Invalid username or password. Please try again.")
                
    elif menu == "Create Account":
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        
        if st.button("Create Account"):
            if new_username in st.session_state.users:
                st.warning("Username already exists. Please choose a different one.")
            elif new_username == "" or new_password == "":
                st.warning("Username and password cannot be empty.")
            else:
                st.session_state.users[new_username] = new_password
                st.success(f"User {new_username} registered successfully!")


#3 main page
else:
    st.title("📊 Linear Regression Model")
    
    # Logout button 
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.users = {}
        st.success("Logged out successfully!")
        
    # Upload file
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) 
        st.write("Data Preview:", df.head())
        st.write("Data Shape:", df.shape)
        
        target = st.selectbox("Select Target Column", df.columns)
        features = st.multiselect("Select Features", df.columns.drop(target))
        if st.button("Train Model"):
            
            if features and target:
                X = df[features]
                y = df[target]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.success("Model Trained Successfully!")
                st.write("Model Coefficients:")
                st.write(model.coef_)
                st.write("Intercept:", model.intercept_)
                st.write("R² Score:", r2_score(y_test, y_pred))
                st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
                st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
                st.write("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
            else:
                st.warning("Please select at least one feature and a target column.")