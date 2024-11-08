from flask import Flask, request, jsonify, send_file   # Response
import pandas as pd
# import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split
from sklearn import tree, ensemble, linear_model, svm
import io
import json
import uuid
import markdown
# import os

app = Flask(__name__)

# In-memory storage
model_classes = {}
fitted_models = {}
datasets = {}
target_scalers = {}
preprocessors = {}


def find_model_class(model_name):
    for module in [tree, ensemble, linear_model, svm]:
        if hasattr(module, model_name):
            return getattr(module, model_name)
    return None


@app.route('/')
def home():
    try:
        with open('README.md', 'r') as f:
            content = f.read()
            return markdown.markdown(content)
    except FileNotFoundError:
        return "README.md not found", 404


@app.route('/test')
def test():
    return jsonify({"message": "Server is running"}), 200


@app.route('/routes')
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "url": str(rule),
            "methods": list(rule.methods)
        })
    return jsonify(routes), 200


@app.route('/add_algorithm', methods=['POST'])
def add_algorithm():
    algorithm_name = request.json.get('algorithm_name')
    if not algorithm_name:
        return jsonify({"error": "Algorithm name is required"}), 400

    model_class = find_model_class(algorithm_name)
    if model_class:
        model_classes[algorithm_name] = model_class
        return jsonify({
            "message": f"Algorithm {algorithm_name} added successfully",
            "available_algorithms": list(model_classes.keys())
        }), 200
    else:
        return jsonify({"error": f"Algorithm {algorithm_name} not found"}), 404


@app.route('/algorithms')
def list_algorithms():
    return jsonify(list(model_classes.keys())), 200


@app.route('/delete_algorithm', methods=['POST'])
def delete_algorithm():
    algorithm_name = request.json.get('algorithm_name')
    if algorithm_name not in model_classes:
        return jsonify({"error": "Algorithm not found"}), 404

    del model_classes[algorithm_name]
    return jsonify({
        "message": f"Algorithm {algorithm_name} deleted successfully",
        "available_algorithms": list(model_classes.keys())
    }), 200


def load_datasets(train_file, test_file=None, index_col=None, header='infer'):
    train_df = pd.read_csv(train_file, index_col=index_col, header=header)
    test_df = pd.read_csv(test_file, index_col=index_col, header=header) if test_file else None
    return train_df, test_df


def preprocess_data(train_df, test_df, target_col, drop_cols, na_strategy, scale_method):
    train_df = train_df.drop(columns=drop_cols, errors='ignore')
    if test_df is not None:
        test_df = test_df.drop(columns=drop_cols, errors='ignore')

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    if test_df is not None:
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
    else:
        X_test, y_test = None, None

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=na_strategy)),
        ('scaler', StandardScaler() if scale_method == 'standard' else MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_train = preprocessor.fit_transform(X_train)
    if X_test is not None:
        X_test = preprocessor.transform(X_test)

    target_scaler = MinMaxScaler()
    y_train = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    if y_test is not None:
        y_test = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    return X_train, y_train, X_test, y_test, target_scaler, preprocessor


@app.route('/train', methods=['POST'])
def train_model():
    if 'train' not in request.files:
        return jsonify({"error": "No train file provided"}), 400

    train_file = request.files['train']
    test_file = request.files.get('test')
    target_col = request.form.get('target_col')
    drop_cols = request.form.get('drop_cols', '').split(',')
    na_strategy = request.form.get('na_strategy', 'mean')
    scale_method = request.form.get('scale_method', 'standard')
    model_type = request.form.get('model_type')
    model_params = json.loads(request.form.get('model_params', '{}'))

    if not model_type or model_type not in model_classes:
        return jsonify({"error": "Invalid or unregistered model type"}), 400

    train_df, test_df = load_datasets(train_file, test_file)
    X_train, y_train, X_test, y_test, target_scaler, preprocessor = preprocess_data(
        train_df, test_df, target_col, drop_cols, na_strategy, scale_method
    )

    model = model_classes[model_type](**model_params)
    model.fit(X_train, y_train)

    model_id = str(uuid.uuid4())
    fitted_models[model_id] = model
    datasets[model_id] = (X_train, y_train, X_test, y_test)
    target_scalers[model_id] = target_scaler
    preprocessors[model_id] = preprocessor

    return jsonify({
        "message": "Model trained successfully",
        "model_id": model_id
    }), 200


@app.route('/models')
def list_models():
    return jsonify(list(fitted_models.keys())), 200


@app.route('/delete_model', methods=['POST'])
def delete_model():
    model_id = request.json.get('model_id')
    if model_id not in fitted_models:
        return jsonify({"error": "Model not found"}), 404

    del fitted_models[model_id]
    del datasets[model_id]
    del target_scalers[model_id]

    return jsonify({"message": f"Model {model_id} deleted successfully"}), 200


@app.route('/predict_by_name', methods=['POST'])
def predict():
    model_id = request.json.get('model_id')
    use_training_data = request.json.get('use_training_data', False)

    if model_id not in fitted_models:
        return jsonify({"error": "Model not found"}), 404

    model = fitted_models[model_id]
    X_train, y_train, X_test, y_test = datasets[model_id]
    target_scaler = target_scalers[model_id]

    if use_training_data or X_test is None:
        X = X_train
        y_true = y_train
    else:
        X = X_test
        y_true = y_test

    y_pred = model.predict(X)
    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

    results = pd.DataFrame({
        'true_values': y_true,
        'predictions': y_pred
    })

    output = io.StringIO()
    results.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'{model_id}_predictions.csv'
    )


if __name__ == '__main__':
    app.run(debug=True)
