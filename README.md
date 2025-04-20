# Resume NER with BERT Fine-Tuning

## Introduction
- End-to-end pipeline to extract structured entities from unstructured resumes
- Utilizes Hugging Face’s `bert-base-uncased` with a custom BIO tagging schema
- Targets HR automation: personal details, education, work history, skills

## Background and Motivation
- Manual resume screening is time-consuming and error-prone
- Automated NER improves consistency, speed, and candidate insight
- BERT’s contextual embeddings excel at token-level classification tasks

## Data Preprocessing
- **Ingest CSV**: load annotated resume tokens with Pandas (`pd.read_csv`)
- **Convert Annotations**: JSON-compatible formatting, flatten nested records
- **BIO Tagging**: add `bio_tag` column indicating entity boundaries
- **Clean & Save**: finalize dataset as `bio_tagged.csv` ready for modeling

## BERT Fine-Tuning
- **Tokenizer**: `BertTokenizerFast` with `is_split_into_words=True`, `max_length=128`
- **Label Alignment**: `tokenize_and_align_labels` maps subwords to BIO tags
- **Dataset**: create Hugging Face `Dataset` splits for train/test
- **Model Setup**: `BertForTokenClassification` with custom `id2label`/`label2id`
- **TrainingArguments**: lr=2e-5, batch_size=8, weight_decay=0.01, epochs=5
- **Trainer**: compute precision, recall, macro-F1 via `compute_metrics`
- **Save Artifacts**: export model and tokenizer to `resume_ner_model`

## Inference and Evaluation
- **Load Model**: `from_pretrained('resume_ner_model')`
- **Predict**: tokenization, forward pass, decode BIO labels
- **Post-process**: merge subwords, reconstruct entities in text
- **Metrics**: classification report (precision, recall, F1) ignoring padding

## Flow of Code
  1. **INIT**: install dependencies, launch Ollama server
  2. **Data Ingestion**: upload and read CSV, initial cleaning
  3. **Preprocessing**: tokenization, label alignment, CSV export
  4. **Model Training**: instantiate model, configure `Trainer`, train
  5. **Evaluation**: compute metrics, display classification report
  6. **Inference Demo**: `predict_entities` function usage
  7. **Presentation Prep**: generate slides and visualizations

## License
- MIT License
- See `LICENSE` for full terms

