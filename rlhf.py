import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from detoxify import Detoxify
import json
import os

# ✅ Step 1: Define a Pretrained Model for Fine-Tuning
MODEL_PATH = "distilbert-base-uncased"

# ✅ Step 2: Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ✅ Step 3: Create a Sample Human Feedback Dataset
sample_data = [
    {"prompt": "What is AI?", "response": "AI is the simulation of human intelligence by machines.", "label": 1},
    {"prompt": "Tell me about machine learning.", "response": "Machine learning is a subset of AI focused on patterns in data.", "label": 1},
    {"prompt": "Can AI be harmful?", "response": "AI can be harmful if misused, especially in surveillance and bias.", "label": 1},
    {"prompt": "Is AI always right?", "response": "No, AI is prone to bias and errors.", "label": 1}
]

dataset_path = os.path.join(os.getcwd(), "human_feedback.json")
with open(dataset_path, "w") as f:
    json.dump(sample_data, f)

# ✅ Step 4: Train the Reward Model for RLHF
def train_reward_model():
    """
    Fine-tunes a reward model on human feedback to align AI with ethical behavior.
    """
    reward_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1)
    dataset = load_dataset("json", data_files=dataset_path)

    # Preprocessing function (Fix: Convert labels to float)
    def preprocess(examples):
        tokenized_inputs = tokenizer(examples["response"], truncation=True, padding="max_length")
        tokenized_inputs["label"] = [float(label) for label in examples["label"]]  # Ensure dtype is float
        return tokenized_inputs

    tokenized_dataset = dataset.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir="reward_model",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=reward_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"]
    )

    trainer.train()
    reward_model.save_pretrained("reward_model")
    print("✅ Reward model training complete and saved to 'reward_model'.")
    return reward_model

# ✅ Step 5: Implement Toxicity Detection
def detect_toxicity(response):
    """
    Uses Detoxify to identify and warn against toxic AI-generated responses.
    """
    try:
        result = Detoxify('original').predict(response)
        if result["toxicity"] > 0.5:
            return "⚠️ Warning: This response may be toxic."
    except Exception as e:
        return f"Error in toxicity detection: {str(e)}"
    return response

# ✅ Step 6: Rank AI Responses Using RLHF
def rank_responses_with_rlhf(reward_model, responses):
    """
    Uses the trained RLHF reward model to rank AI-generated responses.
    """
    scored_responses = []
    for response in responses:
        inputs = tokenizer(response, return_tensors="pt", truncation=True, padding="max_length")
        with torch.no_grad():
            output = reward_model(**inputs)
        score = output.logits.item()
        scored_responses.append((response, score))

    ranked = sorted(scored_responses, key=lambda x: x[1], reverse=True)
    return ranked

# ✅ Step 7: Main Function
def main():
    # Train the Reward Model
    reward_model = train_reward_model()

    # Example AI Responses for Testing Ethical Guardrails
    generated_responses = [
        "AI is great, but it can be biased if not trained well.",
        "All AI is dangerous and should be banned forever!",
        "AI is useful in healthcare, finance, and automation.",
        "AI should replace all human jobs immediately."
    ]

    # Rank Responses Based on Ethics and Safety
    ranked = rank_responses_with_rlhf(reward_model, generated_responses)
    print("\n🔹 **Ranked Responses (Response - Score):**")
    for resp, score in ranked:
        print(f"{score:.4f} - {resp}")

    # Apply Toxicity Detection to the Top-Ranked Response
    best_response = ranked[0][0]
    final_response = detect_toxicity(best_response)

    print("\n✅ **Final Selected Response:**")
    print(final_response)

# ✅ Run the script when executed directly
if __name__ == "__main__":
    main()
