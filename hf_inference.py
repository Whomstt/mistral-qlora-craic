# Import necessary libraries
import mlflow


# Load the fine-tuned model from MLflow (replace with desired run ID)
model_uri = "runs:/0d2c570b1f2a466dbe97e00cf90dfeb4/model"
pipe = mlflow.transformers.load_model(model_uri, device_map="auto")

# Take a prompt and generate text until user exits
while True:
    prompt = input("Enter your prompt: ")
    output = pipe(prompt, do_sample=True, max_new_tokens=30, temperature=0.7, top_p=0.9)
    print(output[0]["generated_text"])
