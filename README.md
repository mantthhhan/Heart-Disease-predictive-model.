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
