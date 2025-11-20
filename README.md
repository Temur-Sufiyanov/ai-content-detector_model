AI Content Detector Model 

A lightweight and fast DistilBERT-based AI Content Detector for classifying text as Human-written or AI-generated.
This repository contains the fine-tuned .pt model, tokenizer configuration, and example usage code for easy integration with FastAPI, Django, Flask, or any Python backend.
Features
  DistilBERT architecture → small, fast, high accuracy
  model format → easy to load in PyTorch
  Optimized for API deployment (FastAPI-ready)
  Supports text classification (AI vs Human)
  Includes example inference script
  Lightweight (good for CPU and low-RAM servers)
Installation:
  pip install torch transformers
Load the Model: 
      import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    
    MODEL_PATH = "model.pt" change model name
    
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    
    def predict_text(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            return "AI-generated" if prediction == 1 else "Human-written"
    
    print(predict_text("This is a sample text."))
Training Overview
  Base model: distilbert-base-uncased
  Trained on a combination of: 
  Custom dataset
  HuggingFace AI/Human datasets  
  Preprocessed Kaggle datasets
Output labels:
  0 = Human
  1 = AI-generated
Use Cases
  Detecting AI-generated essays  
  Verifying authenticity of student assignments
  Checking website/blog content originality
  API integrations for moderation systems
  Research & NLP experiments

Feel free to open issues or request improvements!

