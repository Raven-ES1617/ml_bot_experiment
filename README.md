```markdown
# API Documentation for Machine Learning Model Management and Prediction

This API provides functionality to manage machine learning algorithms, train models, make predictions, and manage datasets. It includes endpoints for loading data, preprocessing, model training, and exporting predictions.

## Available Endpoints

### 1. Home (`/`)
- **Description:** Displays the content of the README file if available, rendered as HTML.
- **Method:** `GET`
- **Sample Request:**
  ```http
  GET /
  ```

### 2. Test Server (`/test`)
- **Description:** Returns a simple message to verify that the server is running.
- **Method:** `GET`
- **Sample Request:**
  ```http
  GET /test
  ```

### 3. List Routes (`/routes`)
- **Description:** Returns a list of all available API routes and supported methods.
- **Method:** `GET`
- **Sample Request:**
  ```http
  GET /routes
  ```

### 4. Add Algorithm (`/add_algorithm`)
- **Description:** Registers a machine learning algorithm for future model training.
- **Method:** `POST`
- **Required JSON Fields:**
  - `algorithm_name` (string): Name of the algorithm (e.g., `"DecisionTreeClassifier"`).
- **Sample Request:**
  ```
  POST /add_algorithm
  {
    "algorithm_name": "DecisionTreeClassifier"
  }
  ```

### 5. List Algorithms (`/algorithms`)
- **Description:** Returns a list of all registered algorithms.
- **Method:** `GET`
- **Sample Request:**
  ```http
  GET /algorithms
  ```

### 6. Delete Algorithm (`/delete_algorithm`)
- **Description:** Deletes a specified algorithm from the registry.
- **Method:** `POST`
- **Required JSON Fields:**
  - `algorithm_name` (string): Name of the algorithm to delete.
- **Sample Request:**
  ```
  POST /delete_algorithm
  {
    "algorithm_name": "DecisionTreeClassifier"
  }
  ```

### 7. Train Model (`/train`)
- **Description:** Trains a specified model with uploaded data.
- **Method:** `POST`
- **Form Data:**
  - `train` (file): CSV file for training data.
  - `test` (optional, file): CSV file for testing data.
  - `target_col` (string): Target column name for prediction.
  - `drop_cols` (string, comma-separated): Columns to drop from the dataset.
  - `na_strategy` (string): Strategy for handling NaNs (`"mean"`, `"median"`, `"most_frequent"`).
  - `scale_method` (string): Scaling method (`"standard"` or `"minmax"`).
  - `model_type` (string): Name of the registered model.
  - `model_params` (JSON): JSON object with parameters for the model.
- **Sample Request:**
  ```http
  POST /train
  Content-Type: multipart/form-data

  train=<train_file.csv>
  test=<test_file.csv>
  target_col="target"
  drop_cols="col1,col2"
  na_strategy="mean"
  scale_method="standard"
  model_type="DecisionTreeClassifier"
  model_params="{}"
  ```

### 8. List Models (`/models`)
- **Description:** Returns a list of all trained models.
- **Method:** `GET`
- **Sample Request:**
  ```http
  GET /models
  ```

### 9. Delete Model (`/delete_model`)
- **Description:** Deletes a specified trained model.
- **Method:** `POST`
- **Required JSON Fields:**
  - `model_id` (string): ID of the model to delete.
- **Sample Request:**
  ```
  POST /delete_model
  {
    "model_id": "model123"
  }
  ```

### 10. Make Prediction (`/predict_by_name`)
- **Description:** Generates predictions using a specified model on either the training or testing dataset.
- **Method:** `POST`
- **Required JSON Fields:**
  - `model_id` (string): ID of the model to use.
  - `use_training_data` (boolean, default: `false`): If `true`, uses training data; otherwise, uses testing data.
- **Sample Request:**
  ```
  POST /predict_by_name
  {
    "model_id": "model123",
    "use_training_data": false
  }
  ```

## Helper Functions

- **`find_model_class(model_name)`**: Searches for the model class by name in scikit-learn modules.
- **`load_datasets(train_file, test_file, index_col, header)`**: Loads and processes CSV datasets for training/testing.
- **`preprocess_data(train_df, test_df, target_col, drop_cols, na_strategy, scale_method)`**: Preprocesses datasets with scaling, encoding, and NaN handling.

Each endpoint returns JSON-formatted responses or files, making it compatible with various applications.
```