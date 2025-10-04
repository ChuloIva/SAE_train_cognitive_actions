meta-llama/Llama-3.1-8B-Instruct is the model I would like to use
It's in my huggingface cache locally

this is kind of the summary of the task

Train a domain-adapted SAE on LLM internal activations using their methodology
Use LLM activations (not external embeddings) as input
Train on domain-specific text 
Keep their smaller M values for interpretability

cognitive_actions_7k_final_1759233061.jsonl

this is the dataset I would like to train my SAE on, check readme of hypotheSAEs to see what the M and K values should be..