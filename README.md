# Heart-Disease-predictive-model.
I have a dataset that contains information about patients, including various features that determine whether they have heart disease or not. 
My objective is to develop a predictive model that can classify patients based on their likelihood of having heart disease. 

To achieve this, I will use the logistic regression classification algorithm, which is well-suited for binary classification problems like this. 

![image](https://github.com/user-attachments/assets/5f136343-9f52-46e9-9a64-03616b4add01)

By training the model on this data, I aim to improve the accuracy of predictions and potentially provide insights into the key factors contributing to heart disease.

![image](https://github.com/user-attachments/assets/cfabc692-2538-4720-abb5-5490e2485b85)

pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['blue','green'])

plt.title('heart disease frequency for sex')

plt.xlabel('sex(0=Female,1=Male)')

plt.xticks(rotation=0)

plt.legend(["havn't disease",'have disease'])

plt.ylabel('frequency')

plt.show()

![image](https://github.com/user-attachments/assets/713569ab-aa6a-4496-a896-f4396864a83b)

This code helps me visualize the relationship between gender and heart disease frequency using my dataset. Here’s what each part does:

pd.crosstab(df.sex, df.target)

I create a table that counts the number of patients based on gender (sex) and whether they have heart disease (target).
In my dataset, sex is coded as 0 for females and 1 for males, while target indicates the presence (1) or absence (0) of heart disease.
.plot(kind="bar", figsize=(15,6), color=['blue','green'])

I generate a bar chart with a size of 15x6 inches.
The bars are colored blue and green to represent patients with and without heart disease.
plt.title('Heart Disease Frequency for Sex')

I set the title to clearly describe what the chart represents.
plt.xlabel('Sex (0 = Female, 1 = Male)')

I label the x-axis to make it clear that 0 stands for females and 1 for males.
plt.xticks(rotation=0)

I keep the x-axis labels horizontal for better readability.
plt.legend(["Haven't Disease", "Have Disease"])

I add a legend to indicate which color represents patients with and without heart disease.
plt.ylabel('Frequency')

I label the y-axis to show that the chart represents the frequency of patients in each category.
plt.show()

Finally, I display the bar chart.
Why I Use This Code: This visualization helps me understand how heart disease is distributed between males and females in my dataset. It allows me to quickly identify patterns, such as whether one gender is more likely to have heart disease.

plt.scatter(x=df.age[df.target==1],y=df.thalach[(df.target==1)],c='red')

plt.scatter(x=df.age[df.target==0],y=df.thalach[(df.target==0)],c='blue')

plt.legend(['disease','not disease'])

plt.xlabel('age')

plt.ylabel('max heart rate')

plt.show()

![image](https://github.com/user-attachments/assets/8a8b3b1d-7854-4a3f-aa9c-28d000ab5c8b)

This code creates a scatter plot to visualize the relationship between age and maximum heart rate, distinguishing between patients with and without heart disease.

plt.scatter(x=df.age[df.target==1], y=df.thalach[df.target==1], c='red')

I plot points where patients have heart disease (target == 1).
The x-axis represents the patients' ages.
The y-axis represents their maximum heart rate (thalach).
I color these points red to differentiate them from others.
plt.scatter(x=df.age[df.target==0], y=df.thalach[df.target==0], c='blue')

I plot points where patients do not have heart disease (target == 0).
I use the same x-axis (age) and y-axis (maximum heart rate).
I color these points blue to distinguish them from those with the disease.
plt.legend(['disease', 'not disease'])

I add a legend to explain that red points represent patients with heart disease, and blue points represent those without it.
plt.xlabel('Age')

I label the x-axis to indicate that it represents the patients' age.
plt.ylabel('Max Heart Rate')

I label the y-axis to show that it represents the maximum recorded heart rate.
plt.show()

Finally, I display the scatter plot.
Why I Use This Code: This visualization helps me understand how age and maximum heart rate relate to heart disease. It allows me to see if there is a noticeable trend, such as whether younger or older patients with high heart rates are more likely to have heart disease.

a= pd.get_dummies(df['cp'],prefix='cp')

b= pd.get_dummies(df['thal'], prefix= "thal")

c= pd.get_dummies(df['slope'], prefix= "slope")

frames=[df,a,b,c]

df= pd.concat(frames, axis=1)

df.head()

df= df.drop(columns=['cp','thal','slope'])

df.head()

y= df.target.values

x= df.drop(['target'],axis=1)

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

lr= LogisticRegression()

lr.fit(x_train,y_train)

y_pred= lr.predict(x_test)

y_pred

y_test

from sklearn.metrics import confusion_matrix

c_m= confusion_matrix(y_test,y_pred)

c_m

accuracy={}

acc= lr.score(x_test,y_test)*100

accuracy['logistic regression']= acc

print("Test Accuracy {:.2f}%".format(acc))

from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors=2)

knn.fit(x_train,y_train)

prediction= knn.predict(x_test)

prediction

print("{} NN Score: {:.2f}%".format(2, knn.score(x_test, y_test)*100))

scorelist=[]

for i in range(1,20) :
    knn2= KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train,y_train)
    scorelist.append(knn2.score(x_test,y_test))

plt.plot(range(1,20),scorelist)

plt.xticks(np.arange(1,20,1))

plt.xlabel(" K values")

plt.ylabel("score")

plt.show()

acc== max(scorelist)*100

accuracy['KNN']=acc

print("Maximum KNN Score is {:.2f}%".format(acc))

![image](https://github.com/user-attachments/assets/49d43c35-4218-465c-8180-51f3064e5314)

from sklearn.svm import SVC

svm = SVC(kernel='linear',random_state = 0)
svm.fit(x_train, y_train)

acc = svm.score(x_test,y_test)*100

accuracy['SVM'] = acc

print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train, y_train)

acc = dtc.score(x_test, y_test)*100

accuracy['Decision Tree'] = acc

print("Decision Tree Test Accuracy {:.2f}%".format(acc))


from sklearn.ensemble import RandomForestClassifier

rfc= RandomForestClassifier(n_estimators=100,criterion='entropy')

rfc.fit(x_train,y_train)

acc= rfc.score(x_test,y_test)*100

accuracy['random forest']=acc

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))

colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=list(accuracy.keys()), y=list(accuracy.values()), palette=colors)
plt.show()

![image](https://github.com/user-attachments/assets/4e6bdb50-ce4d-4df3-bec6-9c3cb1ad56da)

This code performs machine learning classification to predict heart disease using various algorithms and evaluates their accuracy.

1. Data Preprocessing - One-Hot Encoding:
I convert categorical variables (cp, thal, and slope) into dummy variables using pd.get_dummies() to prepare them for machine learning models.
I concatenate these new dummy variables with the original dataset and then drop the original categorical columns.
Splitting Features and Target Variable:

y = df.target.values → I extract the target column as my output variable.
x = df.drop(['target'], axis=1) → I remove target from the dataset to keep only the input features.
Train-Test Split:

I split my dataset into training and testing sets using train_test_split() (80% training, 20% testing).

2. Logistic Regression - Model Training & Prediction:

I create and train a Logistic Regression model using lr.fit(x_train, y_train).
I predict values for x_test using y_pred = lr.predict(x_test).
Performance Evaluation:

I calculate the confusion matrix with confusion_matrix(y_test, y_pred).
I compute the accuracy score and store it in a dictionary (accuracy).

3. K-Nearest Neighbors (KNN) - Model Training & Prediction:

I train a KNN model with n_neighbors=2.
I make predictions and compute the accuracy score.
Finding the Best K Value:

I iterate over different K values (from 1 to 19), train the model for each, and store the accuracy scores.
I plot the accuracy scores against K values to find the optimal K.
I store the maximum KNN accuracy.

4. Support Vector Machine (SVM):
I train an SVM model with a linear kernel and evaluate its accuracy.

5. Decision Tree Classifier:
I train a Decision Tree model and compute its accuracy.

6. Random Forest Classifier:
I train a Random Forest model with 100 trees (n_estimators=100) and entropy as the criterion.
I compute and store its accuracy.

7. Comparing Model Performance:
I visualize the accuracy of all models using a bar chart with seaborn.
The y-axis represents accuracy, and the x-axis represents the different machine learning algorithms.
Different colors are used to differentiate the models.

Final Outcome: After running all models, I can compare their performance and identify the best-performing algorithm for heart disease prediction.
