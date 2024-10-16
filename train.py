import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import plotly.express as px
import plotly.graph_objects as go
import os

df = pd.read_csv("emergingunicorn_companies.csv")
df = df.drop(columns=['Unnamed: 0', 'company_link', 'img_src'])

def fill_missing_lead_investors(group):
    mode_value = group['lead_investors'].mode()
    if not mode_value.empty:
        group['lead_investors'] = group['lead_investors'].fillna(mode_value[0])
    else:
        group['lead_investors'] = group['lead_investors'].fillna('Unknown')
    return group

df_grouped = df.groupby(['country', 'region'], group_keys=False).apply(fill_missing_lead_investors)
df_grouped = df_grouped.reset_index(drop=True)

df['post_money_value'] = df['post_money_value'].replace(r'[\$,M]', '', regex=True).astype(float)
df['total_eq_funding'] = df['total_eq_funding'].replace(r'[\$,M,B]', '', regex=True).astype(float)

sns.histplot(df['post_money_value'], kde=True)
plt.title('Distribution of Post Money Value')
plt.show()

fig = go.Figure()
for investor in df['lead_investors'].unique():
    investor_data = df[df['lead_investors'] == investor]
    fig.add_trace(go.Histogram(
        x=investor_data['post_money_value'],
        name=investor,
        opacity=0.75
    ))

fig.update_layout(
    title='Distribution of Post Money Value by Lead Investors',
    xaxis_title='Post Money Value ($M)',
    yaxis_title='Count',
    barmode='overlay',
    xaxis=dict(tickangle=90),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.05,
        title="Lead Investors",
        traceorder="normal",
        font=dict(size=12),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    )
)
fig.show()

investor_group = df.groupby('lead_investors').agg({
    'post_money_value': 'mean',
    'company_name': lambda x: ', '.join(x.unique())
}).reset_index()

investor_group = investor_group.sort_values(by='post_money_value', ascending=False)

fig = px.bar(
    investor_group,
    x='post_money_value',
    y='lead_investors',
    text='company_name',
    orientation='h',
    title='Average Post Money Value by Lead Investors',
    labels={'post_money_value': 'Average Post Money Value ($M)', 'lead_investors': 'Lead Investors'}
)

fig.update_traces(texttemplate='%{text}', textposition='outside')
fig.update_layout(
    height=3500,
    yaxis=dict(tickangle=0, dtick=1),
    margin=dict(l=200, r=20, t=40, b=20)
)
fig.show()

categorical_features = df.select_dtypes(include=['object']).columns.difference(['post_money_value', 'company_name'])
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(df[categorical_features])
encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
df = pd.concat([df.drop(columns=categorical_features), encoded_features_df], axis=1)

X = df.drop(columns=['post_money_value', 'company_name'])
y = df['post_money_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def log_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    signature = infer_signature(X_test, y_pred)
    print(f'Mean Squared Error ({model_name}): {mse}')
    print(f'R² score ({model_name}): {r2}')
    
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        print(f"Logged {model_name} model with MSE: {mse} and R²: {r2}")
        return run.info.run_id

linear_run_id = log_model(LinearRegression(), "Linear Regression", X_train, y_train, X_test, y_test)
rf_run_id = log_model(RandomForestRegressor(random_state=42), "Random Forest Regressor", X_train, y_train, X_test, y_test)
mlp_run_id = log_model(MLPRegressor(random_state=42, max_iter=1000), "MLP Regressor", X_train, y_train, X_test, y_test)
gb_run_id = log_model(GradientBoostingRegressor(random_state=42), "Gradient Boosting Regressor", X_train, y_train, X_test, y_test)


gradient_br_model = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.05)
gradient_br_model.fit(X_train, y_train)
y_pred = gradient_br_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
input_example = X_test.iloc[:5]
signature = infer_signature(X_test, y_pred)

with mlflow.start_run(run_name="Gradient Boosting Regressor (Tuned)") as run:
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(gradient_br_model, "model", signature=signature, input_example=input_example)
    print(f"Logged tuned Gradient Boosting Regressor model with MSE: {mse} and R²: {r2}")
    tuned_gb_run_id = run.info.run_id

model_uri = f"runs:/{tuned_gb_run_id}/model"
loaded_gradient_br_model = mlflow.sklearn.load_model(model_uri)

print(f"Loaded model from run ID: {tuned_gb_run_id}")

coefficients = pd.DataFrame(LinearRegression().fit(X_train, y_train).coef_, X.columns, columns=['Coefficients'])
sorted_coefficients = coefficients.sort_values(by='Coefficients', ascending=False)
print("Top 20 Positive Impact Investors:")
print(sorted_coefficients.head(20))
print("\nTop 20 Negative Impact Investors:")
print(sorted_coefficients.tail(20))
