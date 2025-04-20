import pickle

with open('data.pkl', 'rb') as file:
    all_data = pickle.load(file)

from sklearn.preprocessing import LabelEncoder

# Get all unique tags
all_tags = set(tag for example in all_data for tag in example['ner_tags'])

label_list = sorted(list(all_tags))
label_encoder = LabelEncoder()
label_encoder.fit(label_list)

# Map tags to IDs
for example in all_data:
    example['label_ids'] = label_encoder.transform(example['ner_tags'])

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example['tokens'], is_split_into_words=True, truncation=True, padding='max_length', max_length=128)

    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)  # Ignored in loss
        elif word_idx != previous_word_idx:
            labels.append(example['label_ids'][word_idx])
        else:
            labels.append(example['label_ids'][word_idx] if example['ner_tags'][word_idx].startswith("I-") else -100)
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

from datasets import Dataset
tokenized_dataset = [tokenize_and_align_labels(example) for example in all_data]

hf_dataset = Dataset.from_list(tokenized_dataset)

dataset = hf_dataset.train_test_split(test_size=0.2)

token = ""

from huggingface_hub import login
login(token=token)  # Will prompt you to paste your HF token

from transformers import BertForTokenClassification

model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_list),
    id2label={i: label for i, label in enumerate(label_list)},
    label2id={label: i for i, label in enumerate(label_list)},
)

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    "bert-resume-ner",
    #evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

results = trainer.evaluate()
print(results)

pip install --upgrade transformers

import os
os.environ["WANDB_DISABLED"] = "true"

import wandb
api_key = ""
wandb.login(key=api_key)  # You’ll be prompted to paste your API key

wandb.init(project="bert-resume-ner")  # Optional: name your project

trainer.train()

trainer.save_model("resume_ner_model")
trainer.model.save_pretrained("resume_ner_model")

from transformers import BertForTokenClassification, BertTokenizer

model = BertForTokenClassification.from_pretrained("resume_ner_model")
tokenizer = BertTokenizer.from_pretrained("resume_ner_model")

import torch
def predict_entities(text):
    model.eval()

    # Move model to same device as tokenizer inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the same device

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [label_list[p.item()] for p in predictions[0]]

    return list(zip(tokens, labels))

predict_entities("Charlie Sharma - Experienced Java developer with Docker and HTML. GoogleSoft (2020-2024) NewYorkpur, Chandashi, USA")

jd = df.iloc[0]['resume_text']

from IPython.display import HTML, display

def set_css():
    display(HTML('''
        <style>
            pre { white-space: pre-wrap; }
        </style>
    '''))

get_ipython().events.register('pre_run_cell', set_css)

"""# PRESENTATION"""

# @markdown # **FINAL**

import json
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# -----------------------------------
# Merge subword tokens only
# -----------------------------------
def merge_entities(predictions):
    """
    Given a list of (token, label) predictions, merge subword tokens (those starting with "##")
    with their immediate previous token. Only merge contiguous subwords, not all tokens with the same label.
    Returns three sets: skills, eligibility, and experience.
    """
    merged_skills = set()
    merged_eligibility = set()
    merged_experience = set()

    prev_token = ""
    prev_label = ""

    for token, label in predictions:
        if token in ["[CLS]", "[SEP]"]:
            continue
        label = label.upper()
        # If token is a subword, merge with the previous token
        if token.startswith("##"):
            prev_token += token[2:]
        else:
            # Finalize previous token (if exists)
            if prev_token and prev_label:
                if "SKILL" in prev_label:
                    merged_skills.add(prev_token.lower())
                elif "QUALIFICATION" in prev_label or "ELIGIBILITY" in prev_label:
                    merged_eligibility.add(prev_token.lower())
                elif "EXPERIENCE" in prev_label:
                    merged_experience.add(prev_token.lower())
            # Start a new token group
            prev_token = token
            prev_label = label
    # Final flush
    if prev_token and prev_label:
        if "SKILL" in prev_label:
            merged_skills.add(prev_token.lower())
        elif "QUALIFICATION" in prev_label or "ELIGIBILITY" in prev_label:
            merged_eligibility.add(prev_token.lower())
        elif "EXPERIENCE" in prev_label:
            merged_experience.add(prev_token.lower())

    return merged_skills, merged_eligibility, merged_experience

# -----------------------------------
# Extract JD entities using NER, after cleaning JD text
# -----------------------------------
def extract_jd_entities_ner(jd_text):
    # Preprocess: remove stopwords & punctuation before running NER
    words = jd_text.split()
    cleaned_words = [
        word.strip(string.punctuation)
        for word in words
        if word.lower() not in stop_words and word.strip(string.punctuation)
    ]
    cleaned_text = " ".join(cleaned_words)

    predictions = predict_entities(cleaned_text)
    return merge_entities(predictions)

# -----------------------------------
# Extract entities from a resume entry (from its tokens & ner_tags)
# -----------------------------------
def extract_resume_entities(entry):
    tokens = entry.get("tokens", [])
    ner_tags = entry.get("ner_tags", [])
    skills = set()
    eligibility = set()
    experience = set()

    prev_token = ""
    prev_label = ""

    for token, label in zip(tokens, ner_tags):
        label = label.upper()
        # Merge subword tokens if token starts with "##"
        if token.startswith("##"):
            prev_token += token[2:]
        else:
            if prev_token and prev_label:
                if "SKILL" in prev_label:
                    skills.add(prev_token.lower())
                elif "QUALIFICATION" in prev_label or "ELIGIBILITY" in prev_label:
                    eligibility.add(prev_token.lower())
                elif "EXPERIENCE" in prev_label:
                    experience.add(prev_token.lower())
            prev_token = token
            prev_label = label
    # Final flush:
    if prev_token and prev_label:
        if "SKILL" in prev_label:
            skills.add(prev_token.lower())
        elif "QUALIFICATION" in prev_label or "ELIGIBILITY" in prev_label:
            eligibility.add(prev_token.lower())
        elif "EXPERIENCE" in prev_label:
            experience.add(prev_token.lower())

    return skills, eligibility, experience

# -----------------------------------
# Compute matching score including experience
# -----------------------------------
def compute_match_score(resume_skills, resume_elig, resume_exp, jd_skills, jd_elig, jd_exp):
    matched_skills = resume_skills.intersection(jd_skills)
    matched_eligibility = resume_elig.intersection(jd_elig)
    matched_experience = resume_exp.intersection(jd_exp)
    score = len(matched_skills) + 2 * len(matched_eligibility) + len(matched_experience)
    return score, matched_skills, matched_eligibility, matched_experience

# -----------------------------------
# Resume matching function
# -----------------------------------
def match_resumes_with_jd(jd_text, data):
    jd_skills, jd_elig, jd_exp = extract_jd_entities_ner(jd_text)

    best_score = -1
    best_final_annotation = None
    best_details = None

    for entry in data:
        resume_skills, resume_elig, resume_exp = extract_resume_entities(entry)
        score, matched_skills, matched_eligibility, matched_experience = compute_match_score(
            resume_skills, resume_elig, resume_exp, jd_skills, jd_elig, jd_exp
        )
        if score > best_score:
            best_score = score
            best_final_annotation = entry.get("final_annotation", "No annotation provided")
            best_details = (matched_skills, matched_eligibility, matched_experience)

    return best_score, best_details, best_final_annotation

# -----------------------------------
# Pretty printing a resume (for demonstration)
# -----------------------------------
def print_resume(data):
    print(f"{'='*60}")
    print(f"📄 Resume: {data.get('Name', 'Unknown')}")
    print(f"{'='*60}\n")

    print("🎓 Qualifications:")
    for qual in data.get("Qualifications", []):
        print(f"  • {qual}")
    print()

    print("🛠️ Skills:")
    for skill in data.get("Skills", []):
        print(f"  • {skill}")
    print()

    print("💼 Experience:")
    for exp in data.get("Experience", []):
        print(f"  • {exp}\n")
    print(f"{'='*60}")

# -----------------------------------
# Sample Job Descriptions to test
# -----------------------------------
# -----------------------------------
# Assuming your resume data is in variable 'data'
# 'data' should be a list of dictionaries, each with keys:
# 'tokens', 'ner_tags', and 'final_annotation' (which is a JSON string of the full resume details)
# For demonstration, here's a dummy example resume data:

# -----------------------------------
# Run Matching for Each JD and Print Results
# -----------------------------------


def predict_entities(text):
    model.eval()

    # Move model to same device as tokenizer inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the same device

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [label_list[p.item()] for p in predictions[0]]

    return list(zip(tokens, labels))



def run(jds):
  for jd in jds:
      print("-----------------------------------------------------")
      print("Job Description:")
      print(jd)
      print()
      data = all_data
      best_score, best_details, final_annotation = match_resumes_with_jd(jd, data)

      print("✅ Best Match Score:", best_score)
      if best_details:
          matched_skills, matched_elig, matched_exp = best_details
          print("🎯 Matched Skills:", sorted(matched_skills))
          print("🎓 Matched Eligibility:", sorted(matched_elig))
          print("💼 Matched Experience:", sorted(matched_exp))
      else:
          print("No matched details.")

      print("\n📎 Final Annotation of Best Matching Resume:")
      try:
          final_annotation_dict = json.loads(final_annotation)
          print_resume(final_annotation_dict)
      except Exception as e:
          print("⚠️ Error parsing final annotation:", e)
          print(final_annotation)

jds = [
    """Software Engineer:
As a Software Engineer, you will be responsible for designing, developing, and maintaining software applications that meet business needs. You will collaborate with cross-functional teams, including product managers, designers, and other engineers, to deliver high-quality solutions. Your role will involve writing clean, efficient, and scalable code, conducting code reviews, and troubleshooting software issues. You will also be expected to stay updated with the latest technologies and best practices in software development.

Requirements:
- Bachelor's degree in Computer Science, Software Engineering, or a related field
- Proficiency in programming languages such as Python, Java, or C++
- Experience with software development lifecycle, including Agile methodologies
- Strong problem-solving skills and ability to debug complex issues
- Knowledge of cloud computing platforms such as AWS, Azure, or Google Cloud
- Excellent communication and teamwork skills""",

    """Data Analyst:
As a Data Analyst, you will be responsible for collecting, processing, and analyzing large datasets to extract meaningful insights that drive business decisions. You will develop reports, dashboards, and visualizations to communicate findings effectively. Your role will involve working closely with stakeholders to understand their data needs and provide actionable recommendations. You will also ensure data integrity and accuracy by implementing data validation techniques.

Requirements:
- Strong analytical and problem-solving skills
- Experience with SQL for querying databases and data manipulation
- Proficiency in data visualization tools such as Tableau, Power BI, or Excel
- Bachelor's degree in Statistics, Mathematics, Data Science, or a related field
- Knowledge of machine learning concepts and predictive analytics is a plus
- Ability to communicate complex data findings in a clear and concise manner""",

    """Cybersecurity Specialist:
As a Cybersecurity Specialist, you will be responsible for implementing security measures to protect systems, networks, and data from cyber threats. You will monitor security incidents, conduct vulnerability assessments, and respond to security breaches. Your role will involve developing security policies, training employees on cybersecurity best practices, and ensuring compliance with industry regulations.

Requirements:
- Knowledge of cybersecurity frameworks such as NIST, ISO 27001, or CIS
- Experience with penetration testing, ethical hacking, and vulnerability assessment
- Certifications such as CISSP, CEH, or CompTIA Security+ preferred
- Strong understanding of encryption, firewalls, and intrusion detection systems
- Bachelor's degree in Cybersecurity, Computer Science, or a related field
- Ability to work under pressure and handle security incidents effectively""",

    """Cloud Engineer:
As a Cloud Engineer, you will design, implement, and manage cloud infrastructure solutions that optimize performance, security, and cost efficiency. You will work with cloud service providers such as AWS, Azure, or Google Cloud to deploy scalable applications. Your role will involve automating cloud operations, monitoring system performance, and troubleshooting issues.

Requirements:
- Experience with cloud computing platforms (AWS, Azure, Google Cloud)
- Knowledge of containerization and orchestration tools such as Docker and Kubernetes
- Strong scripting skills in Python, Bash, or PowerShell
- Bachelor's degree in Computer Science, Cloud Computing, or a related field
- Understanding of cloud security best practices and compliance standards
- Ability to optimize cloud resources for cost efficiency""",

    """DevOps Engineer:
As a DevOps Engineer, you will be responsible for automating deployment processes, improving system reliability, and ensuring seamless collaboration between development and operations teams. You will implement CI/CD pipelines, monitor system performance, and troubleshoot infrastructure issues.

Requirements:
- Experience with CI/CD tools such as Jenkins, GitLab CI/CD, or CircleCI
- Knowledge of scripting languages such as Python, Bash, or Ruby
- Familiarity with Kubernetes, Docker, and cloud infrastructure
- Strong problem-solving skills and ability to optimize deployment workflows
- Bachelor's degree in Computer Science, DevOps, or a related field
- Excellent communication and collaboration skills""",

    """IT Support Specialist:
As an IT Support Specialist, you will provide technical assistance to end-users, troubleshoot hardware and software issues, and ensure smooth IT operations within the organization. You will install and configure computer systems, respond to service requests, and maintain IT documentation.

Requirements:
- Strong problem-solving skills and ability to diagnose technical issues
- Experience with Windows, Linux, and macOS operating systems
- Knowledge of networking fundamentals and troubleshooting techniques
- Excellent communication and customer service skills
- Bachelor's degree in Information Technology or a related field
- Ability to work in a fast-paced environment and prioritize tasks effectively""",

    """Database Administrator:
As a Database Administrator, you will manage and optimize database systems to ensure data integrity, security, and performance. You will be responsible for database backup and recovery, monitoring system health, and implementing best practices for data management.

Requirements:
- Experience with SQL and NoSQL databases such as MySQL, PostgreSQL, MongoDB
- Knowledge of database backup, recovery, and replication strategies
- Strong understanding of indexing, query optimization, and performance tuning
- Bachelor's degree in Information Technology, Database Management, or a related field
- Ability to troubleshoot database issues and implement security measures""",

    """Network Engineer:
As a Network Engineer, you will design, configure, and maintain network infrastructure to ensure seamless connectivity and security. You will monitor network performance, troubleshoot issues, and implement security protocols.

Requirements:
- Experience with routing and switching technologies
- Knowledge of network security protocols such as VPN, firewalls, and IDS/IPS
- Certifications such as CCNA, CCNP, or equivalent preferred
- Strong problem-solving skills and ability to optimize network performance
- Bachelor's degree in Computer Networking or a related field""",

    """AI/ML Engineer:
As an AI/ML Engineer, you will develop machine learning models and AI solutions that enhance business processes and decision-making. You will optimize algorithms for performance and accuracy, work with large datasets, and deploy AI applications.

Requirements:
- Experience with Python and ML frameworks such as TensorFlow, PyTorch, or Scikit-learn
- Strong mathematical and statistical skills
- Bachelor's or Master's degree in AI, Data Science, or a related field
- Knowledge of deep learning, NLP, and computer vision techniques""",

    """UI/UX Designer:
As a UI/UX Designer, you will create user-friendly interfaces and improve user experience through research, design, and usability testing. You will collaborate with developers and stakeholders to ensure intuitive and visually appealing designs.

Requirements:
- Proficiency in design tools such as Figma, Adobe XD, or Sketch
- Experience with front-end technologies like HTML, CSS, and JavaScript
- Strong understanding of user-centered design principles
- Bachelor's degree in Graphic Design, UI/UX, or a related field
- Ability to conduct usability testing and iterate designs based on feedback"""
]

run(jds)

# FINNAL ANNOTATION FROM ANNOTATION COLUMN


# Function to process each row and extract relevant information
def extract_final_annotation(annotation):
    try:
        if pd.isna(annotation):  # Skip empty or NaN annotations
            return None

        annotation = json.loads(annotation)  # Convert string to dictionary

        # Extract fields safely
        # name = annotation.get("name", "").strip()
        name = annotation.get("name", "")
        name = str(name).strip() if name else ""

        # Process qualifications (Ensure it's a list)
        qualifications = annotation.get("qualifications", [])
        if isinstance(qualifications, dict):
            qualifications = [f"{k}: {v}" for k, v in qualifications.items()]
        elif not isinstance(qualifications, list):
            qualifications = [str(qualifications)]  # Ensure it's a list of strings

        # Process skills (Ensure it's a list)
        skills = annotation.get("skills", [])
        if isinstance(skills, dict):
            skills = [f"{k}: {v}" for k, v in skills.items()]
        elif not isinstance(skills, list):
            skills = [str(skills)]  # Ensure it's a list of strings

        # Process experience (Ensure it's a list)
        # Process experience dynamically without hardcoding missing values
        experience_data = annotation.get("experience", [])
        experience = []
        for exp in experience_data:
            if isinstance(exp, dict):
                exp_details = [str(value) for key, value in exp.items() if value]  # Collect non-empty fields
                if exp_details:  # Only add if there's valid data
                    experience.append(", ".join(exp_details))
            else:
                experience.append(str(exp))  # If it's already a string, add as is


        final_annotation_dict = {
            "Name": name,
            "Qualifications": qualifications,
            "Skills": skills,
            "Experience": experience
        }

        return json.dumps(final_annotation_dict, ensure_ascii=False)  # Convert to JSON string

    except json.JSONDecodeError:
        return None  # Skip row if JSON parsing fails

# Apply function to each row and store results in a new column
df["final_annotation"] = df["annotations"].apply(extract_final_annotation)



# Save the updated CSV file
output_path = "/content/final_resume_annotation_procd.csv"
df.to_csv(output_path, index=False)

print(f"Updated file saved at: {output_path}")

df[df['bio_tag'].isna()]

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    true_predictions = []
    true_labels = []

    for pred, label in zip(predictions, labels):
        for p_i, l_i in zip(pred, label):
            if l_i != -100:  # ignore padding
                true_predictions.append(p_i)
                true_labels.append(l_i)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average="macro")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }