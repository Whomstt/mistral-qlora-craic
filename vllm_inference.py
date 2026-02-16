# Import necessary libraries
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


base_model = "Whomstt/Ministral-3-3B-Base-2512-bnb-nf4"
adapter = "Whomstt/mistral-qlora-craic"
llm = LLM(model=base_model, tokenizer=base_model, enable_lora=True)
sampling_params = SamplingParams(max_tokens=30, temperature=0.7, top_p=0.9)

lora_request = LoRARequest("craic-adapter", 1, adapter)

# Take a prompt and generate text until user exits
while True:
    prompt = input("Enter your prompt: ")
    outputs = llm.generate(
        [prompt],
        sampling_params=sampling_params,
        lora_request=lora_request
    )
    print(outputs[0].outputs[0].text)
