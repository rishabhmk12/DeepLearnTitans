import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import gradio as gr

# Load model and tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank for LoRA matrices
    lora_alpha=32,
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
5
# Fine-tuning function
def fine_tune(train_texts, train_labels, epochs=3):
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
    labels = tokenizer(train_labels, truncation=True, padding=True, return_tensors="pt")["input_ids"]
    
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}, self.labels[idx]
    
    train_dataset = Dataset(train_encodings, labels)
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        save_steps=10,
        logging_steps=10,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    return "Fine-tuning complete!"

# RLHF Reward Model
class RewardModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = torch.nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        reward = self.reward_head(outputs.last_hidden_state[:, -1, :])
        return reward

# Gradio UI for LCNC Interaction
def train_pipeline(train_texts, train_labels, epochs):
    result = fine_tune(train_texts.split("\n"), train_labels.split("\n"), int(epochs))
    return result

grammar_ui = gr.Interface(
    fn=train_pipeline,
    inputs=["text", "text", "number"],
    outputs="text",
    title="Low-Code LLM Fine-Tuning & RLHF",
    description="Upload dataset & train a model with RLHF feedback",
)

grammar_ui.launch()