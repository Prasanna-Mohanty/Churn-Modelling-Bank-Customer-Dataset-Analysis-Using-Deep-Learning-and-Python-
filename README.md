---


---

<p><strong>Churn-Modelling Bank Customer Dataset Analysis Using Deep Learning and Python</strong></p>
<p>Source – Udemy/Kaggle</p>
<p>About the Data Set: -</p>
<p>Churn prediction is one of the most popular use cases across all industries. It consists of detecting customers of a Bank who are likely to leave in future. Churn prediction can be extremely useful for customer retention and by predicting in advance customers that are at risk of leaving.</p>
<p>I’ve used Artificial Neural Network ( which is Part of Deep Learning) to complete this analysis. This task is divided into different sections.</p>
<p><strong>Part 1 - Data Preprocessing</strong></p>
<p>To import and manipulate the data I’ve used the most popular packages in Python, Pandas and NumPy. Pandas is a data manipulation package and NumPy is the fundamental package for scientific computing with Python.</p>
<p>Below is the info function of Panda</p>
<p><em><a href="http://dataset.info">dataset.info</a>()</em></p>
<p>&lt;class ‘pandas.core.frame.DataFrame’&gt;</p>
<p>RangeIndex: 10000 entries, 0 to 9999</p>
<p>Data columns (total 14 columns):</p>
<p>RowNumber  10000 non-null int64</p>
<p>CustomerId  10000 non-null int64</p>
<p>Surname  10000 non-null object</p>
<p>CreditScore  10000 non-null int64</p>
<p>Geography  10000 non-null object</p>
<p>Gender  10000 non-null object</p>
<p>Age  10000 non-null int64</p>
<p>Tenure  10000 non-null int64</p>
<p>Balance  10000 non-null float64</p>
<p>NumOfProducts  10000 non-null int64</p>
<p>HasCrCard  10000 non-null int64</p>
<p>IsActiveMember  10000 non-null int64</p>
<p>EstimatedSalary  10000 non-null float64</p>
<p>Exited  10000 non-null int64</p>
<p>dtypes: float64(2), int64(9), object(3)</p>
<p>memory usage: 1.1+ MB</p>
<p>This function tells us that there are 14 columns and 10,000 rows. This dataset contains 13 independent variables and one dependent variable (“Exited” column).<br>
Now let’s separate dataset into X(independent variables) and y(dependent variable). In X we’ll keep from third column to 13th (since we don’t need “RowNumber”,”CustomerId” and “Exited”) and in y we only need “Exited”  field.</p>
<p><strong>Encoding Categorical Data :-</strong></p>
<p>We have two categorical variables (“Country” and “Gender” variable) in our data. We need to encode them. I’ve used labelencoder and onehotencoder of scikit-learn package to achieve this. <strong>Note that I’ve removed a dummy variable of “Country” to avoid falling into dummy variable trap.</strong></p>
<p><strong>Feature scaling:</strong></p>
<p>Feature scaling is an important part in machine learning and deep learning to ease the calculations.</p>
<p>Now our data is well preprocessed and now we will build the artificial neural network.</p>
<p><strong>Part 2:- Building the ANN</strong></p>
<p>To start with building the ANN we will first import Tensorflow and Kreas Library.</p>
<p><em>import keras</em><br>
<em>from keras.models import Sequential</em><br>
<em>from keras.layers import Dense</em></p>
<p>The sequential module is required to initialize the ann and dense module is required to add layers to it.<br>
Next, we will initialize the deep learning model as a sequence of layers</p>
<p><em>classifier=Sequential()</em></p>
<p>Next, we will add the input layer and hidden layer to the model.</p>
<h4 id="adding-layers"><strong>Adding Layers:</strong></h4>
<p><em>#first hidden layer</em><br>
<em>classifier.add(Dense(units=6,kernel_initializer=’uniform’,activation=’relu’,input_dim=11))</em></p>
<p>I have defined 6 units in the first hidden layer and used rectifying linear function (ReLu) as the activation function. And since input dimensions (no. of features) are 11 we will define input_dim=11.</p>
<p>Next, we will add second hidden layer</p>
<p><em>classifier.add(Dense(units=6,kernel_initializer=’uniform’,activation=’relu’))</em></p>
<h3 id="adding-the-output-layer"><strong>Adding the output layer:</strong></h3>
<p><em>classifier.add(Dense(units=1,kernel_initializer=’uniform’,activation=’sigmoid’))</em></p>
<p>It will have one output unit and we will use sigmoid function since it’s a binary classification.</p>
<p>Now since we have defined the model of the neural network, we will compile the model.</p>
<p><em>classifier.compile(optimizer=’adam’,loss=’binary_crossentropy’,metrics=[‘accuracy’])</em></p>
<h3 id="fitting-the-training-data"><strong>Fitting the training data:</strong></h3>
<p>Now to train the neural network we have to fit the training data or feed the train data to the network .We can do this by using fit function.</p>
<p><em>classifier.fit(X_train,y_train,batch_size=10,epochs=100)</em></p>
<p>Now our neural network will learn from the training data by using forward propagation, gradient descent and backward propagation.</p>
<h3 id="part-3---prediction"><strong>Part 3: -</strong> <strong>Prediction</strong></h3>
<p>Now the final step is to predict that test data and evaluating the model.</p>
<p>This confusion Matrix will validate our data model and tell us how well our model works on the data provided to it.</p>
<p>Also, I’ve merged the test data and predicted result into one CSV file.</p>

