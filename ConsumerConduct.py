from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

main = tkinter.Tk()
main.title("Consumer COnduct")
main.geometry("1300x1200")

global le1, le2, le3, le4, cls, extension_acc

def upload():
    global filename
    global data
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "Dataset")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")

def importdata():
    global filename
    global df
    df = pd.read_csv(filename,encoding = 'latin1')
    text.insert(END,"Data Information:\n"+str(df.head())+"\n")
    text.insert(END,"Columns Information:\n"+str(df.columns)+"\n")
    
def preprocess():
    global df
    global x,y
    X = df.drop(columns=['Clicked on Ad'])
    label_names = np.array(['No','Yes'])
    y = df['Clicked on Ad'].values
    feature_names = np.array(list(X))
    x = np.array(X)
    sns.countplot(df["Clicked on Ad"])
    plt.show()

def plotCorrelationMatrix(df, graphWidth):
    #filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        text.insert(END,f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.show()

def ttmodel():
    global le1, le2, le3, le4
    global x,y
    global df
    global X_train,X_test,y_train,y_test
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=5)
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    le4 = LabelEncoder()

    x[:,4] = le1.fit_transform(x[:,4])
    x[:,5] = le2.fit_transform(x[:,5])
    x[:,7] = le3.fit_transform(x[:,7])
    x[:,8] = le4.fit_transform(x[:,8])
    
    X_train[:,4] = le1.fit_transform(X_train[:,4])
    X_train[:,5] = le2.fit_transform(X_train[:,5])
    X_train[:,7] = le3.fit_transform(X_train[:,7])
    X_train[:,8] = le4.fit_transform(X_train[:,8])

    X_test[:,4] = le1.fit_transform(X_test[:,4])
    X_test[:,5] = le2.fit_transform(X_test[:,5])
    X_test[:,7] = le3.fit_transform(X_test[:,7])

    X_test[:,8] = le4.fit_transform(X_test[:,8])
    
    text.insert(END,"Train Shape: "+str(X_train.shape)+"\n")
    text.insert(END,"Test Shape: "+str(X_test.shape)+"\n")

    plotCorrelationMatrix(df, len(df.columns))

def mlmodels():
    global x,y
    global X_train,X_test,y_train,y_test,cls, extension_acc
    global lr_acc,svc_acc,rfc_acc,gnb_acc,dtc_acc
    clf_lr = LogisticRegression(random_state=0)
    clf_lr.fit(X_train,y_train)
    pred = clf_lr.predict(X_test)
    lr_acc=clf_lr.score(X_test, y_test)
    text.insert(END,"LOGIT Accuracy: "+str(clf_lr.score(X_test, y_test))+"\n")
    text.insert(END,"LOGIT recall_score: "+str(recall_score(y_test,pred))+"\n")
    text.insert(END,"LOGIT precision_score: "+str(precision_score(y_test,pred))+"\n")
    text.insert(END,"LOGIT f1_score: "+str(f1_score(y_test,pred))+"\n\n")

    clf_svc = LinearSVC(random_state=0)
    clf_svc.fit(X_train,y_train)
    clf_svc.score(X_test,y_test)
    pred = clf_svc.predict(X_test)
    svc_acc=clf_svc.score(X_test, y_test)
    text.insert(END,"SVC Accuracy: "+str(clf_svc.score(X_test, y_test))+"\n")
    text.insert(END,"SVC recall_score: "+str(recall_score(y_test,pred))+"\n")
    text.insert(END,"SVC precision_score: "+str(precision_score(y_test,pred))+"\n")
    text.insert(END,"SVC f1_score: "+str(f1_score(y_test,pred))+"\n\n")

    clf_gnb = GaussianNB()
    clf_gnb.fit(X_train,y_train)
    clf_gnb.score(X_test,y_test)
    pred = clf_gnb.predict(X_test)
    gnb_acc=clf_gnb.score(X_test, y_test)
    text.insert(END,"Naive Bayes Accuracy: "+str(clf_gnb.score(X_test, y_test))+"\n")
    text.insert(END,"Naive Bayes recall_score: "+str(recall_score(y_test,pred))+"\n")
    text.insert(END,"Naive Bayes precision_score: "+str(precision_score(y_test,pred))+"\n")
    text.insert(END,"Naive Bayes f1_score: "+str(f1_score(y_test,pred))+"\n\n")

    clf_rfc = RandomForestClassifier(random_state=0)
    clf_rfc.fit(X_train,y_train)
    clf_rfc.score(X_test,y_test)
    pred = clf_rfc.predict(X_test)
    rfc_acc=clf_rfc.score(X_test, y_test)
    text.insert(END,"Random Forest Accuracy: "+str(clf_gnb.score(X_test, y_test))+"\n")
    text.insert(END,"Random Forest recall_score: "+str(recall_score(y_test,pred))+"\n")
    text.insert(END,"Random Forest precision_score: "+str(precision_score(y_test,pred))+"\n")
    text.insert(END,"Random Forest f1_score: "+str(f1_score(y_test,pred))+"\n\n")

    clf_dtc = DecisionTreeClassifier(random_state=0)
    clf_dtc.fit(X_train,y_train)
    clf_dtc.score(X_test,y_test)
    pred = clf_dtc.predict(X_test)
    dtc_acc=clf_rfc.score(X_test, y_test)
    text.insert(END,"Decision Tree Accuracy: "+str(clf_dtc.score(X_test, y_test))+"\n")
    text.insert(END,"Decision Tree recall_score: "+str(recall_score(y_test,pred))+"\n")
    text.insert(END,"Decision Tree precision_score: "+str(precision_score(y_test,pred))+"\n")
    text.insert(END,"Decision Tree f1_score: "+str(f1_score(y_test,pred))+"\n\n")

    cls = BaggingClassifier()
    cls.fit(x,y)
    cls.score(X_test,y_test)
    pred = cls.predict(X_test)
    extension_acc =cls.score(X_test, y_test)
    text.insert(END,"Extension Bagging Classifier Accuracy: "+str(extension_acc)+"\n")
    text.insert(END,"Extension Bagging Classifier recall_score: "+str(recall_score(y_test,pred))+"\n")
    text.insert(END,"Extension Bagging Classifier precision_score: "+str(precision_score(y_test,pred))+"\n")
    text.insert(END,"Extension Bagging Classifier f1_score: "+str(f1_score(y_test,pred))+"\n\n")

def predict():
    global cls
    global le1, le2, le3, le4
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename,encoding='latin1')
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    XX = dataset[:,0:dataset.shape[1]]
    XX[:,4] = le1.fit_transform(XX[:,4])
    XX[:,5] = le2.fit_transform(XX[:,5])
    XX[:,7] = le3.fit_transform(XX[:,7])
    XX[:,8] = le4.fit_transform(XX[:,8])

    prediction = cls.predict(XX)
    print(prediction)
    for i in range(len(prediction)):
        if prediction[i] == 0:
            text.insert(END,"Test DATA : "+str(dataset[i])+" ===> PREDICTED AS CONSUMER NOT CLICKED ON ADD\n\n")
        if prediction[i] == 1:
            text.insert(END,"Test DATA : "+str(dataset[i])+" ===> PREDICTED AS CONSUMER CLICKED ON ADD\n\n")    

def graph():
    global lr_acc,svc_acc,rfc_acc,gnb_acc,dtc_acc, extension_acc
    
    height = [lr_acc,svc_acc,rfc_acc,gnb_acc,dtc_acc, extension_acc]
    bars = ('Logit', 'SVC','RFC','GNB','DT', 'Extension BaggingClassifier')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()   
                
font = ('times', 16, 'bold')
title = Label(main, text='A Model for prediction of consumer conduct using machine learning algorithm')
title.config(bg='dark salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=upload)
upload.place(x=900,y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=900,y=150)

ip = Button(main, text="Data Import", command=importdata)
ip.place(x=900,y=200)
ip.config(font=font1)

pp = Button(main, text="Data Preprocessing", command=preprocess)
pp.place(x=900,y=250)
pp.config(font=font1)

tt = Button(main, text="Train and Test Model", command=ttmodel)
tt.place(x=900,y=300)
tt.config(font=font1)

ml = Button(main, text="Run Algorithms", command=mlmodels)
ml.place(x=900,y=350)
ml.config(font=font1)

gph = Button(main, text="Accuracy Graph", command=graph)
gph.place(x=900,y=400)
gph.config(font=font1)

predictButton = Button(main, text="Predict Consumer Conduct from Test Data", command=predict)
predictButton.place(x=900,y=450)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='peach puff')
main.mainloop()



