# Prediction of reimboursement loan application for Home Credit customers

Interactive dashboard: https://home-credit-dash.herokuapp.com/
Dataset: https://www.kaggle.com/c/home-credit-default-risk/data

Context/Scenario: You are a Data Scientist at Home Credit, which offers consumer credit to people with little or no loan history. The company wants to develop a scoring model of the customer's probability of default to support the decision of whether or not to grant a loan to a potential customer. In addition, customer relationship managers have pointed out that customers are increasingly demanding transparency in credit decisions.

Problem: Home Credit wants to improve loan application transparency and facilitate the loan application for all clients

Methods:
1. Merge the datasets
2. Highlight imbalance classification problem
4. Feature engineering with log and aggregate functions essentially
5. Bayesian optimisation
6. SKFold
7. Train LightGBM Classifier with AUC, F1 score and custom score as loss function
8. Threshold the results to lower the False Positive
9. Extract the model in a pickle
10. Realize the web portal with HTML5/CSS3, Plotly/Dash and Flask to serve the layouts
11. Present the application loan results to clients with SHAP values and compare to other clients

Results:
1. Custom function to train LightGBM presents the best results for Home Credit financial health 
2. Confusion matrix (TP=238k, TN=18k, FP=7k, FN=45k)
3. Web portal: https://home-credit-dash.herokuapp.com/

Libraries: Pandas, Numpy, Matplotlib, Scikit-learn, LightGBM, hyperopt, SHAP, Dash/Plotly, Flask
