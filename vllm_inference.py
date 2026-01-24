# Import necessary libraries
from vllm import LLM, SamplingParams


# Load the fine-tuned model from MLflow (replace with desired run ID)
model_uri = "runs:/e1c534eaec4a45ad89d9e22af5abbf8e/model"
llm = LLM(model=model_uri)

# Take a prompt and generate text until user exits
while True:
    prompt = input("Enter your prompt: ")
    outputs = llm.generate(
        [prompt],
        sampling_params=SamplingParams(max_tokens=30, temperature=0.7, top_p=0.9),
    )
    print(outputs[0].outputs[0].text)
