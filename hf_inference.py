# Import necessary libraries
import mlflow


# Load the fine-tuned model from MLflow
model_uri = "runs:/2ca73cbd1ce04ed6baa626661ad37a68/model"
pipe = mlflow.transformers.load_model(model_uri, device_map="auto")

# Take a prompt and generate text until user exits
while True:
    prompt = input("Enter your prompt: ")
    output = pipe(prompt, max_new_tokens=30, do_sample=True, temperature=0.7, top_p=0.9)
    print(output[0]["generated_text"])
