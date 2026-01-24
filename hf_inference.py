# Import necessary libraries
import mlflow


# Load the fine-tuned model from MLflow (replace with desired run ID)
model_uri = "runs:/e1c534eaec4a45ad89d9e22af5abbf8e/model"
pipe = mlflow.transformers.load_model(model_uri, device_map="auto")

# Take a prompt and generate text until user exits
while True:
    prompt = input("Enter your prompt: ")
    output = pipe(prompt, max_new_tokens=30, temperature=0.7, top_p=0.9)
    print(output[0]["generated_text"])
