import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Student Pass/Fail Prediction Using Machine Learning")

np.random.seed(42)

n=200

hours=np.random.randint(1,10,n)
attendance=np.random.randint(40,100,n)
assignment=np.random.randint(4,10,n)
midterm=np.random.randint(3,10,n)
final=np.random.randint(3,10,n)

result=((hours+assignment+midterm+final)/4+attendance/50>6).astype(int)

data=pd.DataFrame({
"Hours_Study":hours,
"Attendance":attendance,
"Assignment":assignment,
"Midterm":midterm,
"Final":final,
"Result":result
})

X=data.drop("Result",axis=1)
y=data["Result"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

log=LogisticRegression()
tree=DecisionTreeClassifier()
rf=RandomForestClassifier()

log.fit(X_train,y_train)
tree.fit(X_train,y_train)
rf.fit(X_train,y_train)

pred=rf.predict(X_test)
acc=accuracy_score(y_test,pred)

regX=data[["Hours_Study","Attendance","Assignment","Midterm"]]
regy=data["Final"]

reg=LinearRegression()
reg.fit(regX,regy)
pred_final=reg.predict(regX)

st.subheader("Accuracy Random Forest")
st.write(acc)

st.subheader("Dataset")
st.dataframe(data)

st.subheader("Prediction")

hours_i=st.slider("Hours Study",0,10,5)
attendance_i=st.slider("Attendance",0,100,70)
assignment_i=st.slider("Assignment",0,10,6)
midterm_i=st.slider("Midterm",0,10,6)
final_i=st.slider("Final",0,10,6)

input_data=[[hours_i,attendance_i,assignment_i,midterm_i,final_i]]

if st.button("Predict"):
    p=rf.predict(input_data)
    if p[0]==1:
        st.success("PASS")
    else:
        st.error("FAIL")

st.subheader("Study Hours vs Final Score")

fig,ax=plt.subplots()
ax.scatter(data["Hours_Study"],data["Final"])
ax.set_xlabel("Hours Study")
ax.set_ylabel("Final Score")

st.pyplot(fig)

st.subheader("Pass Fail Distribution")

fig2,ax2=plt.subplots()
data["Result"].value_counts().plot(kind="bar",ax=ax2)
st.pyplot(fig2)

st.subheader("Regression Prediction of Final Score")

df_reg=pd.DataFrame({
"Actual":regy,
"Predicted":pred_final
})

st.dataframe(df_reg)