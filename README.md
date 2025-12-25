<h1 align="center">💳 Credit Card Customer Behaviour Prediction</h1>

<p align="center">
<b>End-to-End Machine Learning Project | Flask Deployment</b>
</p>

<hr>

<h2>📌 Project Overview</h2>
<p>
This project predicts whether a credit card customer is a 
<b>Good</b> or <b>Bad</b> customer using Machine Learning.
It covers the complete lifecycle from data preprocessing to deployment.
</p>

<hr>

<h2>🎯 Business Objective</h2>
<ul>
<li>Identify risky credit card customers</li>
<li>Reduce financial losses</li>
<li>Improve credit approval decisions</li>
</ul>

<hr>

<h2>🗂 Project Structure</h2>

<pre>
creditcard.csv
main.py
random_sample.py
var_out.py
feature_selection.py
data_balance.py
model_training.py
log_code.py
app.py
index.html
requirements.txt
Procfile
credit_card.pkl
scalar.pkl
</pre>

<hr>

<h2>🔄 Machine Learning Pipeline</h2>

<h3>1️⃣ Data Cleaning</h3>
<ul>
<li>Removed invalid target values</li>
<li>Handled incorrect data types</li>
<li>Dropped unnecessary columns</li>
</ul>

<h3>2️⃣ Missing Value Treatment</h3>
<p>Random Sample Imputation to preserve data distribution.</p>

<h3>3️⃣ Feature Transformation & Outliers</h3>
<ul>
<li>Yeo-Johnson transformation</li>
<li>IQR based trimming</li>
</ul>

<h3>4️⃣ Feature Selection</h3>
<ul>
<li>Constant & quasi-constant removal</li>
<li>Pearson correlation based filtering</li>
</ul>

<h3>5️⃣ Encoding & Scaling</h3>
<ul>
<li>One-Hot & Ordinal Encoding</li>
<li>StandardScaler applied</li>
</ul>

<h3>6️⃣ Class Imbalance Handling</h3>
<p>SMOTE applied to balance Good & Bad classes.</p>

<h3>7️⃣ Model Training</h3>
<ul>
<li>KNN</li>
<li>Logistic Regression</li>
<li>Naive Bayes</li>
<li>Decision Tree</li>
<li>Random Forest</li>
<li>AdaBoost</li>
<li><b>Gradient Boosting (Final Model)</b></li>
<li>XGBoost</li>
</ul>

<hr>

<h2>🚀 Deployment</h2>
<ul>
<li>Flask backend</li>
<li>HTML + Bootstrap frontend</li>
<li>Real-time predictions</li>
</ul>

<hr>

<h2>⚙️ How to Run</h2>

<pre>
pip install -r requirements.txt
python main.py
python app.py
</pre>

<p>Open browser: <b>http://127.0.0.1:5000</b></p>

<hr>

<h2>🧠 Technologies Used</h2>
<p>
Python | Pandas | NumPy | Scikit-learn | SMOTE | XGBoost | Flask | HTML
</p>

<hr>

<h2>👤 Author</h2>
<p>
<b>Bala Venu</b><br>
Data Scientist
</p>
