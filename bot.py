import json

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import requests
import os
import io

# Constants
API_URL = "http://127.0.0.1:5000"  # Assuming the Flask API runs locally on port 5000
FILE_PATH = os.path.join(os.getcwd(), "uploads")  # Set the correct directory path

# Ensure the directory exists
os.makedirs(FILE_PATH, exist_ok=True)

uploaded_files = {}  # Dictionary to store user-uploaded files


# Command for starting the bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to ModelMasterBot! Here are the available commands:\n\n"
        "📄 *File Uploads*:\n"
        "/upload_train_data - Upload your training data CSV file (provide command and file in one message)\n"
        "/upload_test_data - Upload your testing data CSV file (provide command and file in one message)\n\n"

        "🔧 *Algorithm Management*:\n"
        "/add_algorithm <algorithm_name> - Register a new algorithm by name\n"
        "/list_algorithms - List all registered algorithms\n"
        "/delete_algorithm <algorithm_name> - Delete a registered algorithm\n\n"

        "📊 *Model Training & Prediction*:\n"
        "/train <target_col> <model_type> - Train a model with uploaded train/test data\n"
        "/list_models - List all trained models\n"
        "/predict <model_id> <use_training_data> - Make a prediction with a model\n"
        "/delete_model <model_id> - Delete a trained model by its ID\n\n"

        "💡 *Help*:\n"
        "/help - Display this help message"
    )


# Command to upload training data
async def upload_train_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await save_uploaded_file(update, context, "train")


# Command to upload test data
async def upload_test_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await save_uploaded_file(update, context, "test")


# Helper function to save uploaded files
async def save_uploaded_file(update: Update, context: ContextTypes.DEFAULT_TYPE, file_type: str):
    user_id = update.message.from_user.id
    file = update.message.document
    file_path = os.path.join(FILE_PATH, f"{user_id}_{file_type}.csv")

    # Download and save the file
    new_file = await context.bot.get_file(file.file_id)
    await new_file.download_to_drive(file_path)

    # Store the file path and confirm the upload
    uploaded_files[user_id] = file_path
    await update.message.reply_text(f"{file_type.capitalize()} data uploaded successfully.")


# Command to add a new algorithm
async def add_algorithm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        algorithm_name = context.args[0]  # Expects algorithm name as argument
        response = requests.post(f"{API_URL}/add_algorithm", json={"algorithm_name": algorithm_name})
        data = response.json()
        await update.message.reply_text(data.get("message", "Failed to add algorithm."))
    except IndexError:
        await update.message.reply_text("Please provide the algorithm name, e.g., /add_algorithm RandomForestRegressor")


# Command to list all algorithms
async def list_algorithms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = requests.get(f"{API_URL}/algorithms")
    data = response.json()
    if data:
        await update.message.reply_text("Registered algorithms:\n" + "\n".join(data))
    else:
        await update.message.reply_text("No algorithms registered yet.")


# Command to delete an algorithm
async def delete_algorithm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        algorithm_name = context.args[0]
        response = requests.post(f"{API_URL}/delete_algorithm", json={"algorithm_name": algorithm_name})
        data = response.json()
        await update.message.reply_text(data.get("message", "Failed to delete algorithm."))
    except IndexError:
        await update.message.reply_text("Please provide the algorithm name to delete.")


# Command to train a model
async def train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id

    if len(context.args) < 2:
        await update.message.reply_text(
            "Please provide the necessary arguments in the format:\n"
            "/train <target_col> <model_type>\nExample: /train price RandomForestRegressor"
        )
        return

    target_col = context.args[0]
    model_type = context.args[1]

    # Check if the user uploaded a file
    file_path = uploaded_files.get(user_id)
    if not file_path:
        await update.message.reply_text(
            "No training file uploaded. Please upload a CSV file before running the /train command.")
        return

    try:
        with open(file_path, 'rb') as train_file:
            files = {'train': train_file}
            form_data = {
                'target_col': target_col,
                'model_type': model_type,
                'drop_cols': [],
                'na_strategy': 'mean',
                'scale_method': 'standard'
            }
            response = requests.post(f"{API_URL}/train", files=files, data=form_data)
            data = response.json()
            await update.message.reply_text(data.get("message", "Failed to train model."))
    except FileNotFoundError:
        await update.message.reply_text("Training data file not found. Please upload the file or specify its location.")
    except json.decoder.JSONDecodeError:
        await update.message.reply_text("Error query, probably no such column in dataset.")


# Command to list all trained models
async def list_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = requests.get(f"{API_URL}/models")
    data = response.json()
    if data:
        await update.message.reply_text("Trained models:\n" + "\n".join(data))
    else:
        await update.message.reply_text("No models trained yet.")


# Command to make a prediction
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        model_id = context.args[0]
        response = requests.post(f"{API_URL}/predict_by_name", json={"model_id": model_id, "use_training_data": False})

        # Check if the response is successful
        if response.status_code == 200:
            # Create a BytesIO stream from the content of the response
            output_stream = io.BytesIO(response.content)
            output_stream.seek(0)  # Move to the beginning of the stream

            # Send the file back to the user as an attachment
            await update.message.reply_document(
                document=output_stream,
                filename=f'{model_id}_predictions.csv',
                caption="Here are your predictions."
            )
        else:
            await update.message.reply_text("Failed to retrieve predictions. Please try again later.")

    except IndexError:
        await update.message.reply_text("Please provide the model ID to make a prediction.")
    except Exception as e:
        await update.message.reply_text(f"An error occurred: {str(e)}")


# Command to delete a model by ID
async def delete_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        model_id = context.args[0]  # Get the model ID from the command arguments
        response = requests.post(f"{API_URL}/delete_model", json={"model_id": model_id})
        data = response.json()
        await update.message.reply_text(data.get("message", "Failed to delete model."))
    except IndexError:
        await update.message.reply_text("Please provide the model ID to delete, e.g., /delete_model <model_id>.")


# Command for help message (reusing the start message)
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to ModelMasterBot! Here are the available commands:\n\n"
        "📄 *File Uploads*:\n"
        "/upload_train_data - Upload your training data CSV file (provide command and file in one message)\n"
        "/upload_test_data - Upload your testing data CSV file (provide command and file in one message)\n\n"

        "🔧 *Algorithm Management*:\n"
        "/add_algorithm <algorithm_name> - Register a new algorithm by name\n"
        "/list_algorithms - List all registered algorithms\n"
        "/delete_algorithm <algorithm_name> - Delete a registered algorithm\n\n"

        "📊 *Model Training & Prediction*:\n"
        "/train <target_col> <model_type> - Train a model with uploaded train/test data\n"
        "/list_models - List all trained models\n"
        "/predict <model_id> <use_training_data> - Make a prediction with a model\n"
        "/delete_model <model_id> - Delete a trained model by its ID\n\n"

        "💡 *Help*:\n"
        "/help - Display this help message"
    )


# Set up the bot with correct handlers
def main():
    app = Application.builder().token('***').build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))  # Register the new help_command handler
    app.add_handler(CommandHandler("upload_train_data", upload_train_data))
    app.add_handler(CommandHandler("upload_test_data", upload_test_data))
    app.add_handler(CommandHandler("add_algorithm", add_algorithm))
    app.add_handler(CommandHandler("list_algorithms", list_algorithms))
    app.add_handler(CommandHandler("delete_algorithm", delete_algorithm))
    app.add_handler(CommandHandler("train", train))
    app.add_handler(CommandHandler("list_models", list_models))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("delete_model", delete_model))  # Register the new delete_model handler

    # Update this line to use a custom handler for document uploads
    app.add_handler(MessageHandler(
        filters.Document.ALL,
        lambda update, context: save_uploaded_file(update,
                                                   context,
                                                   "train" if "train" in update.message.caption else "test")))

    # Start the bot
    app.run_polling()


if __name__ == '__main__':
    main()
