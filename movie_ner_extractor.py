import os
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm
from textblob import TextBlob
import re

nlp = spacy.load("en_core_web_sm")

COMMENTS_FOLDER = "./comments"
MASTER_LIST_PATH = "./tmdb_master_list.csv"
OUTPUT_PATH = "./ner_movie_recommendations.csv"
SUMMARY_PATH = "./ner_summary_by_tier.csv"
TIER_CHART_PATH = "./tier_distribution_chart.png"


def normalize_title(title):
    title = os.path.splitext(title)[0]
    title = title.lower()
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"[^a-z0-9 ]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


MASTER_LIST_PATH = "./tmdb_master_list.csv"
df_master["Normalized Title"] = df_master["Movie Title"].astype(str).apply(normalize_title)
valid_titles = set(df_master["Normalized Title"])

recommendations = []
rejected_titles = defaultdict(lambda: {"Count": 0, "Reason": ""})

comment_files = [f for f in os.listdir(COMMENTS_FOLDER) if f.endswith(".txt")]
total_files = len(comment_files)
print(f"🔄 Starting processing of {total_files} files...")

watched_titles = {normalize_title(f) for f in comment_files}

for filename in tqdm(comment_files, desc="Processing comments", unit="file"):
    watched_title = normalize_title(filename)
    with open(os.path.join(COMMENTS_FOLDER, filename), "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        sentiment_score = TextBlob(sentence_text).sentiment.polarity

        if sentiment_score > 0.2:
            sent_doc = nlp(sentence_text)
            for ent in sent_doc.ents:
                if ent.label_ == "WORK_OF_ART":
                    candidate = normalize_title(ent.text)
                    if candidate and candidate != watched_title and candidate in valid_titles:
                        tier = (
    "Viral" if sentiment_score > 0.85 else
    "Hot" if sentiment_score > 0.6 else
    "Trending" if sentiment_score > 0.4 else
    "Cult" if 0.2 < sentiment_score <= 0.4 else
    "Other"
)
                        recommendations.append({
                            "Movie Title": candidate.title(),
                            "Source File": filename.replace(".txt", ""),
                            "Mentions": 1,
                            "Context": sentence_text,
                            "Full Comment": sentence_text,
                            "Already Watched": "Yes" if candidate in watched_titles else "No",
                            "Tier": tier
                        })
                    else:
                        reason = "Already Watched" if candidate == watched_title else ("Not in Master List" if candidate not in valid_titles else "Unknown")
                        rejected_titles[candidate]["Count"] += 1
                        rejected_titles[candidate]["Reason"] = reason

print("\n⚠️ Top rejected titles (not matched to master list):")
rejected_df = pd.DataFrame([
    {"Rejected Title": k, "Count": v["Count"], "Reason": v["Reason"]} for k, v in sorted(rejected_titles.items(), key=lambda x: x[1]["Count"], reverse=True)
])
rejected_df.to_csv("rejected_titles_log.csv", index=False)

# Suggest additions to master list based on high rejection count
suggested_additions = rejected_df[(rejected_df["Reason"] == "Not in Master List") & (rejected_df["Count"] >= 3)]
suggested_additions[["Rejected Title", "Count"]].to_csv("suggested_master_list_additions.csv", index=False)
print("Top 10 shown below. Full log saved to rejected_titles_log.csv")
print("Suggestions for master list additions saved to suggested_master_list_additions.csv")
for title, data in sorted(rejected_titles.items(), key=lambda x: x[1]["Count"], reverse=True)[:10]:
        print(f"- {title} ({data['Count']} times) - Reason: {data['Reason']}")

counter = Counter((r["Movie Title"], r["Source File"]) for r in recommendations)
aggregated_rows = []
for (movie, source), count in counter.items():
    match = next(r for r in recommendations if r["Movie Title"] == movie and r["Source File"] == source)
    aggregated_rows.append({
        "Movie Title": movie,
        "Source File": source,
        "Mentions": count,
        "Context Example": match["Context"],
        "Full Comment": match["Full Comment"],
        "Already Watched": match["Already Watched"],
        "Tier": match["Tier"],
        "Avg Sentiment": round(TextBlob(match["Full Comment"]).sentiment.polarity, 3)
    })


df_output = pd.DataFrame(aggregated_rows).sort_values(by="Mentions", ascending=False)
df_output.to_csv(OUTPUT_PATH, index=False)

tier_summary = df_output["Tier"].value_counts().reset_index()
tier_summary.columns = ["Tier", "Count"]
tier_summary.to_csv(SUMMARY_PATH, index=False)

plt.figure(figsize=(6, 4))
colors = {"Viral": "purple", "Hot": "red", "Trending": "orange", "Cult": "blue", "Positive": "green", "Other": "gray"}
bar_colors = [colors.get(tier, "gray") for tier in tier_summary["Tier"]]
plt.bar(tier_summary["Tier"], tier_summary["Count"], color=bar_colors)
plt.title("Recommendation Tier Distribution")
plt.xlabel("Tier")
plt.ylabel("Number of Recommendations")
plt.tight_layout()
plt.savefig(TIER_CHART_PATH)
plt.close()

print(f"\n✅ Extracted {len(df_output)} entries across {total_files} files.")
print(f"📄 Output saved to: {OUTPUT_PATH}")
print(f"📊 Tier summary saved to: {SUMMARY_PATH}")
print(f"🖼️ Tier chart saved to: {TIER_CHART_PATH}")