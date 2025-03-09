<body>
  <h1>Heart Disease Prediction</h1>

  <h2>Overview</h2>
  <p>This project implements a machine learning model to predict the presence of heart disease using the Heart Disease Dataset. Various algorithms, including Logistic Regression, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Decision Tree, and Naive Bayes, are utilized to achieve the best accuracy.</p>

  <h2>Dataset</h2>
  <p>The dataset used in this project is the <a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset">Heart Disease Dataset</a>. This dataset contains 76 attributes, but the experiments in this project refer to a subset of 14 attributes. The "target" field indicates the presence of heart disease in the patient, where 0 = no disease and 1 = disease.</p>

  <h3>Attribute Information</h3>
  <ul>
      <li>age</li>
      <li>sex</li>
      <li>chest pain type (4 values)</li>
      <li>resting blood pressure</li>
      <li>serum cholesterol in mg/dl</li>
      <li>fasting blood sugar > 120 mg/dl</li>
      <li>resting electrocardiographic results (values 0,1,2)</li>
      <li>maximum heart rate achieved</li>
      <li>exercise induced angina</li>
      <li>oldpeak = ST depression induced by exercise relative to rest</li>
      <li>the slope of the peak exercise ST segment</li>
      <li>number of major vessels (0-3) colored by fluoroscopy</li>
      <li>thal: 0 = normal; 1 = fixed defect; 2 = reversible defect</li>
      <li>target</li>
  </ul>

  <h2>Requirements</h2>
  <p>To run this project, you need the following libraries:</p>
  <ul>
      <li>Pandas</li>
      <li>NumPy</li>
      <li>Matplotlib</li>
      <li>Seaborn</li>
      <li>Scikit-learn</li>
  </ul>
  <p>You can install the required libraries using pip:</p>
  <pre><code>pip install pandas numpy matplotlib seaborn scikit-learn</code></pre>

  <h2>Installation</h2>
  <ol>
      <li><strong>Clone the Repository</strong> (if applicable):
          <pre><code>git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction</code></pre>
      </li>
      <li><strong>Run the Jupyter Notebook or Python Script</strong>:
          <pre><code>jupyter notebook</code></pre>
          or
          <pre><code>python heart_disease_prediction.py</code></pre>
      </li>
  </ol>

  <h2>Code Explanation</h2>
  <p>The main components of the code are as follows:</p>
  <ul>
      <li><strong>Data Loading:</strong> The heart disease dataset is loaded using Pandas.</li>
      <li><strong>Data Preprocessing:</strong> The target variable is encoded, and the dataset is split into training and testing sets.</li>
      <li><strong>Model Training:</strong> Various models are trained, including Logistic Regression, Random Forest, SVM, KNN, Decision Tree, and Naive Bayes.</li>
      <li><strong>Model Evaluation:</strong> The accuracy, classification report, and confusion matrix are generated for each model to assess performance.</li>
  </ul>

  <h2>Model Performance</h2>
  <p>The following models were evaluated:</p>
  <ul>
      <li><strong>Logistic Regression:</strong> Achieved an accuracy of approximately 78.54%.</li>
      <li><strong>Random Forest:</strong> Achieved an accuracy of approximately 98.54%.</li>
      <li><strong>Support Vector Machine (SVM):</strong> Achieved an accuracy of approximately 68.29%.</li>
      <li><strong>K-Nearest Neighbors (KNN):</strong> Achieved an accuracy of approximately 73.17%.</li>
      <li><strong>Decision Tree:</strong> Achieved an accuracy of approximately 98.54%.</li>
      <li><strong>Naive Bayes:</strong> Achieved an accuracy of approximately 80.00%.</li>
  </ul>

  <h2>Confusion Matrix</h2>
  <p>The confusion matrix provides insight into the model's performance by showing the true positive, true negative, false positive, and false negative rates. Below is an example of the confusion matrix for the Random Forest model:</p>
  <pre><code>[[71 31]
[13 90]]</code></pre>

  <h2>Conclusion</h2>
  <p>This project demonstrates the application of various machine learning algorithms to predict heart disease. The Random Forest and Decision Tree models performed exceptionally well, achieving high accuracy rates. Future work could involve hyperparameter tuning and exploring additional features to improve model performance further.</p>

  <h2>Acknowledgments</h2>
  <ul>
      <li>Thanks to the Kaggle community for providing the dataset.</li>
      <li>Special thanks to the Scikit-learn documentation for guidance on model implementation.</li>
  </ul>

  <h2>License</h2>
  <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
</body>
