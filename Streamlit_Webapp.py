import streamlit as st


# EDA Pkgs
import pandas as pd
import numpy as np


# Data Viz Pkgs
import matplotlib
matplotlib.use('Agg')# To Prevent Errors
import matplotlib.pyplot as plt
import seaborn as sns 


# ML Pkgs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# Basic preprocessing required for all the models.  
def preprocessing(df):
    
    df = df.drop('Unnamed: 32', axis=1)
    
    cols = ['radius_worst', 
         'texture_worst', 
         'radius_mean',
         'perimeter_worst', 
         'compactness_mean',
         'symmetry_mean',
         'fractal_dimension_mean',
         'fractal_dimension_se',
         'radius_worst',
         'radius_se',
         'perimeter_se',
         'smoothness_se',
         'perimeter_mean',
         'smoothness_mean',
         'area_worst',
         'compactness_worst',
         'compactness_se',
         'concave points_worst']
    df = df.drop(cols,axis=1)
    


    cols = [
        'concave points_mean', 
        'concave points_se']
    df = df.drop(cols, axis=1)
    
    # Assign x and y
    x = df.iloc[:,2:].values
    y = df.iloc[:,1].values
    
    
    
    
    return x, y



# Training Decission Tree for Classification
@st.cache(allow_output_mutation=True)
def decisionTree(x_train, x_test, y_train, y_test):
    # Train the model
    dtc = DecisionTreeClassifier(criterion ='entropy', random_state = 1)
    dtc.fit(x_train, y_train)
    y_pred = dtc.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    return score, report, dtc



# Training KNN Classifier
@st.cache(allow_output_mutation=True)
def Knn_Classifier(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    return score, report, knn    



# Training Random Forest for Classification
@st.cache(allow_output_mutation=True)
def randomForest(x_train, x_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators = 10, criterion ='entropy', random_state=1)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    return score, report, rfc



# Training Logistic Regression
@st.cache(allow_output_mutation=True)
def logisticRegression(x_train, x_test, y_train, y_test):
    log = LogisticRegression(random_state=1)
    log.fit(x_train,y_train)
    y_pred = log.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    return score, report, log



# Training Support Vector Machine
@st.cache(allow_output_mutation=True)
def svm(x_train, x_test, y_train, y_test):
    svc_linear = SVC(kernel = 'linear', random_state = 1)
    svc_linear.fit(x_train, y_train)
    y_pred = svc_linear.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    return score, report, svc_linear



@st.cache(allow_output_mutation=True)
def neuralNet(x_train, x_test, y_train, y_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(11,2),activation='relu', random_state=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    
    return score, report, clf


def accept_user_data():
    
    area_mean = st.number_input("Enter the area_mean: ")
    texture_mean = st.number_input("Enter the texture_mean ")
    concavity_mean = st.number_input("Enter the concavity_mean: ")
    texture_se = st.number_input("Enter the texture_se: ")
    area_se = st.number_input("Enter the area_se: ")
    concavity_se = st.number_input("Enter the concavity_se: ")
    symmetry_se = st.number_input("Enter the symmetry_se: ")
    smoothness_worst = st.number_input("Enter the smoothness_worst: ")
    concavity_worst = st.number_input("Enter the concavity_worst: ")
    symmetry_worst = st.number_input("Enter the symmetry_worst: ")
    fractal_dimension_worst = st.number_input("Enter the fractal_dimension_worst: ")
    results = [texture_mean,area_mean,concavity_mean,texture_se,area_se,concavity_se,symmetry_se,smoothness_worst,concavity_worst,symmetry_worst,fractal_dimension_worst]
    
    prettified_result = {"texture_mean":texture_mean,
                         "area_mean":area_mean,
                         "concavity_mean":concavity_mean,
                         "texture_se":texture_se,
                         "area_se":area_se,
                         "concavity_se":concavity_se,
                         "symmetry_se":symmetry_se,
                         "smoothness_worst":smoothness_worst,
                         "concavity_worst":concavity_worst,
                         "symmetry_worst":symmetry_worst,
                         "fractal_dimension_worst":fractal_dimension_worst}
    
    sample_data = np.array(results).reshape(1,-1)
    
    return sample_data,prettified_result,results


def main():
    st.title("Automated Breast Cancer Prediction!")
    st.subheader("Predicting Breast Cancer with ML and Streamlit")
    
    df = pd.read_csv("Breast_Cancer.csv")
    x,y = preprocessing(df)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 1. Splitting x,y into Training & Test set.
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101) 

    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    choose_eda = st.sidebar.selectbox("Select EDA",["NONE","Exploratory Data Analysis"])
    
    if(choose_eda == "Exploratory Data Analysis"):
        
        if st.checkbox("Show DataSet"):
            st.dataframe(df.head())
        
        if st.button("Columns Names"):
            st.write(df.columns)
    
        if st.checkbox("Shape of Dataset"):
            st.write(df.shape)
            data_dim = st.radio("Show Dimension by",("Rows","Columns"))
            if data_dim == 'Rows':
                st.text("Number of  Rows")
                st.write(df.shape[0])
            elif data_dim == 'Columns':
                st.text("Number of Columns")
                st.write(df.shape[1])
    
        if st.checkbox("Select Columns To Show"):
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect('Select',all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)
        
        if st.button("Data Types"):
            st.write(df.dtypes)
        
        if st.button("Value Counts"):
            st.text("Value Counts By Target/Class")
            st.write(df.iloc[:,1].value_counts())
        
        st.subheader("Data Visualization")
        # Show Correlation Plots

        
        # Seaborn Plot
        if st.checkbox("Correlation Plot with Annotation[Seaborn]"):
            corr = df.corr().round(2)
            msk = np.zeros_like(corr, dtype=np.bool)
            msk[np.triu_indices_from(msk)] = True
            f, ax = plt.subplots(figsize=(20,20))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            st.write(sns.heatmap(corr, mask=msk, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True))
            plt.tight_layout()
            st.pyplot()
            
        # Count Plot for Diagnosis
        if st.checkbox("Count Plot for Diagnosis column"):
            st.write(sns.countplot(df['diagnosis'], palette='RdBu'))
            st.pyplot()

    
    choose_pred = st.sidebar.selectbox("Make a Prediction",["NONE","User Inputted Prediction"])
    
    if(choose_pred == "User Inputted Prediction"):
        
        sample_data,prettified_result,results=accept_user_data()
        
        if st.checkbox("Your Inputs Summary"):
            st.json(prettified_result)
            st.text("Vectorized as ::{}".format(results))
        
        st.subheader("Prediction")
        if st.checkbox("Make Prediction"):
            choose_mdl = st.selectbox("Choose a Model:",["Decision Tree","Neural Network","K-Nearest Neighbours","SVM","Logistic Regression","Random Forest Classification"])
            
            if(choose_mdl =="Decision Tree"):
                score, report, dtc = decisionTree(x_train, x_test, y_train, y_test)
                pred = dtc.predict(sc.transform(sample_data))
                st.write("The Predicted Class is: ", le.inverse_transform(pred))
                
            
            elif(choose_mdl == "Neural Network"):
                score,report,clf= neuralNet(x_train,x_test,y_train,y_test)
                pred = clf.predict(sc.transform(sample_data))
                st.write("The Predicted Class is: ", le.inverse_transform(pred))
            
            elif(choose_mdl == "K-Nearest Neighbours"):
                score,report,knn = Knn_Classifier(x_train, x_test, y_train, y_test)
                pred = knn.predict(sc.transform(sample_data))
                st.write("The Predicted Class is: ", le.inverse_transform(pred))
                
            elif(choose_mdl == "SVM"):
                score,report,svc_linear = svm(x_train, x_test, y_train, y_test)
                pred = svc_linear.predict(sc.transform(sample_data))
                st.write("The Predicted Class is: ", le.inverse_transform(pred))
                
            elif(choose_mdl == "Logistic Regression"):
                score,report,log = logisticRegression(x_train, x_test, y_train, y_test)
                pred = log.predict(sc.transform(sample_data))
                st.write("The Predicted Class is: ", le.inverse_transform(pred))
            
            elif(choose_mdl == "Random Forest Classification"):
                score,report,rfc = randomForest(x_train, x_test, y_train, y_test)
                pred = rfc.predict(sc.transform(sample_data))
                st.write("The Predicted Class is: ", le.inverse_transform(pred))

    
    # ML Model Report
    
    
    choose_model = st.sidebar.selectbox("ML Model Analysis",
	["NONE","Decision Tree","Neural Network","K-Nearest Neighbours","SVM","Logistic Regression","Random Forest Classification"])


    if(choose_model == "Decision Tree"):
        st.write("""
# Explore different classifiers
Which one is the best?
""")
        score, report, dtc = decisionTree(x_train, x_test, y_train, y_test)
        st.text("Accuracy of Decision Tree model is: ")
        st.write(score,"%")
        print('\n')
        st.text("Report of Decision Tree model is: ")
        st.write(report)
        print('\n')
        cm = confusion_matrix(y_test,dtc.predict(x_test))
        st.write(sns.heatmap(cm,annot=True,fmt="d", cmap="mako"))
        st.pyplot()
        
    elif(choose_model == "Neural Network"):
        st.write("""
# Explore different classifiers
Which one is the best?
""")
        score,report,clf= neuralNet(x_train,x_test,y_train,y_test)
        st.text("Accuracy of Neural Network model is: ")
        st.write(score,"%")
        print('\n')
        st.text("Report of Neural Network model is: ")
        st.write(report)
        print('\n')
        cm = confusion_matrix(y_test,clf.predict(x_test))
        st.write(sns.heatmap(cm,annot=True,fmt="d", cmap="mako"))
        st.pyplot()
        

    elif(choose_model == "K-Nearest Neighbours"):
        st.write("""
# Explore different classifiers
Which one is the best?
""")
        score,report,knn = Knn_Classifier(x_train, x_test, y_train, y_test)
        st.text("Accuracy of K-Nearest Neighbour model is: ")
        st.write(score,"%")
        print('\n')
        st.text("Report of K-Nearest Neighbour model is: ")
        st.write(report)
        print('\n')
        cm = confusion_matrix(y_test,knn.predict(x_test))
        st.write(sns.heatmap(cm,annot=True,fmt="d", cmap="mako"))
        st.pyplot()

    elif(choose_model == "SVM"):
        st.write("""
# Explore different classifiers
Which one is the best?
""")
        score,report,svc_linear = svm(x_train, x_test, y_train, y_test)
        st.text("Accuracy of SVM model is:")
        st.write(score,"%")
        print('\n')
        st.text("Report of SVM model is:")
        st.write(report)
        print('\n')
        cm = confusion_matrix(y_test,svc_linear.predict(x_test))
        st.write(sns.heatmap(cm,annot=True,fmt="d", cmap="mako"))
        st.pyplot()


    elif(choose_model == "Logistic Regression"):
        st.write("""
# Explore different classifiers
Which one is the best?
""")
        score,report,log = logisticRegression(x_train, x_test, y_train, y_test)
        st.text("Accuracy of Logistic Regression model is: ")
        st.write(score,"%")
        print('\n')
        st.text("Report of Logistic Regression model is: ")
        st.write(report)
        print('\n')
        cm = confusion_matrix(y_test,log.predict(x_test))
        st.write(sns.heatmap(cm,annot=True,fmt="d", cmap="mako"))
        st.pyplot()



    elif(choose_model == "Random Forest Classification"):
        st.write("""
# Explore different classifiers
Which one is the best?
""")
        score,report,rfc = randomForest(x_train, x_test, y_train, y_test)
        st.text("Accuracy of Random Forest Classification model is: ")
        st.write(score,"%")
        print('\n')
        st.text("Report of Random Forest Classification model is: ")
        st.write(report)
        print('\n')
        cm = confusion_matrix(y_test,rfc.predict(x_test))
        st.write(sns.heatmap(cm,annot=True,fmt="d", cmap="mako"))
        st.pyplot()


if __name__ == "__main__":
    main()












    
    
    
    
    
    
    
    
    