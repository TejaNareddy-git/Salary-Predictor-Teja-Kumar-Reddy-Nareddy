# Salary-Predictor-Teja-Kumar-Reddy-Nareddy

💼 **SmartSalary Predictor (Powered by LightGBM)**

An interactive machine learning web app to predict employee salary class using census data.

🚀 Overview
SmartSalary Predictor is an easy-to-use web application that predicts whether an individual's salary is greater than 50K or less than or equal to 50K based on key features such as age, education, occupation, and work hours. It leverages the LightGBM algorithm for robust, accurate classification and is deployed using Streamlit for instant interactivity.

🏗️ Project Structure
text
Salary-Prediction-Project/
├── salary_app.py                 # Streamlit web application code
├── salary_predictor_reduced.pkl  # Trained LightGBM model
├── input_features.pkl            # List of model input feature names
├── label_encoders.pkl            # (Optional) LabelEncoders for categorical mapping
├── salary.csv                    # Original or cleaned training dataset
└── README.md                     # (This) project documentation
📝 Features
Modern Streamlit UI for easy interaction.

Robust LightGBM classifier for salary prediction.

Human-readable dropdowns: Select occupation, workclass, marital status, and nationality naturally.

Confidence Display: See model's certainty for each prediction.

Instant feedback: Visual results with interpretation cues.

Expandable "How it works" section for end-user guidance.

⚙️ System Requirements
Python 3.8 – 3.11

pip or conda for package management

Chrome / Firefox / Edge (for accessing the app)

4GB RAM minimum recommended

📚 Installation
Clone this repository:

git clone https://github.com/yourusername/salary-prediction-project.git
cd salary-prediction-project
Create and activate a virtual environment (recommended):

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

pip install -r requirements.txt
🏁 Quick Start
Train the model (optional, if not using provided .pkl):
Use salary.csv and the provided training scripts to retrain as needed.

Run the web app:

streamlit run salary_app.py
Usage:

Fill in employee features (age, sex, occupation, etc.).

Click “Predict Salary” to view the predicted salary class and confidence.

Use the "How it works" expander for help.

🛠️ Dependencies
pandas

numpy

scikit-learn

lightgbm

joblib

streamlit

matplotlib

seaborn

Install them all at once with:

pip install -r requirements.txt
📊 Model Details
Algorithm: LightGBM (Gradient Boosted Trees)

Selected Features:

age, workclass, education-num, marital-status, occupation, sex, hours-per-week, native-country

Performance:

High accuracy reported on the held-out test set (see evaluation in notebook/scripts).

📂 Training and Data Preparation
Data sourced from census/adult dataset.

Categorical features encoded via scikit-learn LabelEncoder.

Missing values (?) handled and dropped for model quality.

See included EDA and model training notebook (.ipynb) for step-by-step details.

🌐 Deployment
The app runs locally via Streamlit (streamlit run salary_app.py).

For public sharing, deploy to Streamlit Community Cloud, Heroku, or other PaaS.

All .pkl files needed for prediction must be in the app root directory.

🔮 Future Scope
Add batch (CSV) prediction.

Explainable AI: feature importance via SHAP/LIME.

Cloud hosting for one-click sharing.

Improved mobile-responsive UI and dark mode.

🤝 Contributing
Contributions and improvements very welcome!

Fork the repo

Create a feature branch

Submit a pull request

📧 Contact & Acknowledgements
Created by Teja Kumar Reddy Nareddy.
For questions or suggestions, open an issue or email tejntkr@gmail.com.

Empowering data-driven salary insights, one prediction at a time.
