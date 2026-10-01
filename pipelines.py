import pandas as pd
from transformers import pipeline


# ============================================================
# WEEK 7 - HUGGING FACE PIPELINES
# CLASS ACTIVITY + HOME ACTIVITY
# ============================================================


# ------------------------------------------------------------
# 1. INPUT TEXT
# ------------------------------------------------------------

article = """
Artificial Intelligence is becoming an important technology in modern healthcare.
Hospitals and healthcare organizations are using AI to improve patient care,
medical diagnosis, and hospital management.

Machine learning systems can analyze large amounts of medical data and identify
patterns that may be difficult for humans to notice.

One important application of AI in healthcare is disease prediction. AI models
can study patient information such as medical history, symptoms, laboratory
results, and medical images. Based on these patterns, machine learning models
can help doctors identify possible diseases at an early stage.

AI is also being used in medical imaging. Deep learning models can analyze
X-rays, CT scans, and other medical images. These systems can help identify
abnormalities and provide additional information to healthcare professionals.

AI can also support personalized patient care. AI can analyze individual
patient information and help healthcare professionals understand different
treatment requirements.

Despite these benefits, healthcare AI has several challenges. Patient data is
sensitive, so privacy and security must be maintained. AI models also need
high-quality and reliable datasets.

Human supervision is very important when AI is used in healthcare. Doctors and
healthcare professionals should verify important AI-generated results before
making medical decisions.

With responsible development, AI can support disease prediction, medical
research, patient monitoring, and personalized care.
"""


# ------------------------------------------------------------
# 2. SAVE INPUT TEXT
# ------------------------------------------------------------

input_df = pd.DataFrame({
    "text": [article]
})

input_df.to_csv(
    "dataset/input_text.csv",
    index=False
)

print("Input text saved successfully.")


# ============================================================
# CLASS ACTIVITY 1
# SENTIMENT ANALYSIS
# ============================================================

print("\n====================================")
print("1. SENTIMENT ANALYSIS")
print("====================================")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

sentiment_result = sentiment_pipeline(article[:512])

print("Sentiment Result:")
print(sentiment_result)

sentiment_df = pd.DataFrame(sentiment_result)

sentiment_df.to_csv(
    "dataset/sentiment_results.csv",
    index=False
)

print("Sentiment result saved.")


# ============================================================
# CLASS ACTIVITY 2
# TEXT SUMMARIZATION
# ============================================================

print("\n====================================")
print("2. TEXT SUMMARIZATION")
print("====================================")

summarizer = pipeline(
    "summarization",
    model= "google-t5/t5-small"
)

summary_result = summarizer(
    article,
    max_length=100,
    min_length=30,
    do_sample=False
)

summary_text = summary_result[0]["summary_text"]

print("\nGenerated Summary:")
print(summary_text)

summary_df = pd.DataFrame({
    "original_text": [article],
    "summary": [summary_text]
})

summary_df.to_csv(
    "dataset/summarized_text.csv",
    index=False
)

print("Summary saved.")


# ============================================================
# CLASS ACTIVITY 3
# TRANSLATION
# ============================================================

print("\n====================================")
print("3. TRANSLATION")
print("====================================")

# English to French

translator_fr = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-fr"
)

french_result = translator_fr(
    article[:1000]
)

french_text = french_result[0]["translation_text"]

print("\nFrench Translation:")
print(french_text)


# English to German

translator_de = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-de"
)

german_result = translator_de(
    article[:1000]
)

german_text = german_result[0]["translation_text"]

print("\nGerman Translation:")
print(german_text)


translation_df = pd.DataFrame({
    "original_text": [article[:1000]],
    "french_translation": [french_text],
    "german_translation": [german_text]
})

translation_df.to_csv(
    "dataset/translated_text.csv",
    index=False
)

print("Translation results saved.")


# ============================================================
# CLASS ACTIVITY 4
# QUESTION ANSWERING
# ============================================================

print("\n====================================")
print("4. QUESTION ANSWERING")
print("====================================")

qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)


questions = [
    "What is one important application of AI in healthcare?",
    "What type of data can AI analyze?",
    "How can AI help with medical imaging?",
    "Why is human supervision important?",
    "What is necessary for safe healthcare AI?"
]


qa_results = []

for question in questions:

    result = qa_pipeline(
        question=question,
        context=article
    )

    print("\nQuestion:")
    print(question)

    print("Answer:")
    print(result["answer"])

    qa_results.append({
        "question": question,
        "answer": result["answer"],
        "score": result["score"]
    })


qa_df = pd.DataFrame(qa_results)

qa_df.to_csv(
    "dataset/qa_results.csv",
    index=False
)

print("\nQuestion answering results saved.")


# ============================================================
# HOME ACTIVITY 1
# SUMMARY COMPARISON
# ============================================================

print("\n====================================")
print("5. HOME ACTIVITY - SUMMARY COMPARISON")
print("====================================")


summary_60 = summarizer(
    article,
    max_length=60,
    min_length=20,
    do_sample=False
)[0]["summary_text"]


summary_100 = summarizer(
    article,
    max_length=100,
    min_length=30,
    do_sample=False
)[0]["summary_text"]


summary_150 = summarizer(
    article,
    max_length=150,
    min_length=40,
    do_sample=False
)[0]["summary_text"]


print("\nSummary with max_length = 60:")
print(summary_60)

print("\nSummary with max_length = 100:")
print(summary_100)

print("\nSummary with max_length = 150:")
print(summary_150)


comparison_df = pd.DataFrame({
    "type": [
        "Original Text",
        "Summary - max_length 60",
        "Summary - max_length 100",
        "Summary - max_length 150"
    ],

    "output": [
        article,
        summary_60,
        summary_100,
        summary_150
    ]
})


comparison_df.to_csv(
    "dataset/pipeline_comparison.csv",
    index=False
)

print("\nSummary comparison saved.")


# ============================================================
# HOME ACTIVITY 2
# TRANSLATION COMPARISON
# ============================================================

print("\n====================================")
print("6. HOME ACTIVITY - TRANSLATION COMPARISON")
print("====================================")

print("\nOriginal Language: English")

print("\nFrench Output:")
print(french_text)

print("\nGerman Output:")
print(german_text)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n====================================")
print("ALL TASKS COMPLETED SUCCESSFULLY")
print("====================================")

print("\nGenerated Files:")

print("1. dataset/input_text.csv")
print("2. dataset/sentiment_results.csv")
print("3. dataset/summarized_text.csv")
print("4. dataset/translated_text.csv")
print("5. dataset/qa_results.csv")
print("6. dataset/pipeline_comparison.csv")