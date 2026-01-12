# Import necessary libraries
from vllm import LLM, SamplingParams

# Load the base quantized model with LoRA adapter
llm = LLM(model="ministral-3-3b-base-4bit", adapter="mistral-qlora-craic")

# Take a prompt and generate text
while True:
    prompt = input("Enter your prompt: ")
    output = llm.generate([prompt], sampling_params=SamplingParams(max_tokens=30))
    print(output.generations[0].text)
