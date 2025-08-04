"""
Cleaned-up and modularized movie recommendation extractor using NLP and sentiment analysis.
"""

import os
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm
from textblob import TextBlob

# --- Constants ---
MASTER_LIST_PATH = "./tmdb_master_list.csv"
COMMENTS_FOLDER = "./comments"
OUTPUT_RECOMMENDATIONS = "ner_movie_recommendations.csv"
OUTPUT_SUMMARY = "ner_summary_by_tier.csv"
OUTPUT_REJECTIONS = "rejected_titles_log.csv"
OUTPUT_SUGGESTIONS = "suggested_master_list_additions.csv"
OUTPUT_CHART = "tier_distribution_chart.png"
APPROVED_FILE = "approved_master_additions.csv"

# --- Initialization ---
nlp = spacy.load("en_core_web_sm")

# --- Utility Functions ---
def normalize_title(title):
    title = os.path.splitext(title)[0].lower()
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"[^a-z0-9 ]", "", title)
    return re.sub(r"\s+", " ", title).strip()

def get_sentiment_tier(score):
    if score > 0.85:
        return "Viral"
    elif score > 0.6:
        return "Hot"
    elif score > 0.4:
        return "Trending"
    elif score > 0.2:
        return "Cult"
    return "Other"

# --- Load master list ---
df_master = pd.read_csv(MASTER_LIST_PATH)
df_master["Normalized Title"] = df_master["Movie Title"].astype(str).apply(normalize_title)
valid_titles = set(df_master["Normalized Title"])
print("✅ Loaded and normalized TMDb master list")

# --- Process comment files ---
recommendations = []
rejected_titles = defaultdict(lambda: {"Count": 0, "Reason": ""})

comment_files = [f for f in os.listdir(COMMENTS_FOLDER) if f.endswith(".txt")]
watched_titles = {normalize_title(f) for f in comment_files}

print(f"🔄 Starting processing of {len(comment_files)} files...")

for filename in tqdm(comment_files, desc="Processing comments", unit="file"):
    watched_title = normalize_title(filename)
    with open(os.path.join(COMMENTS_FOLDER, filename), "r", encoding="utf-8") as file:
        text = file.read()

    for sent in nlp(text).sents:
        sentence = sent.text.strip()
        sentiment = TextBlob(sentence).sentiment.polarity

        if sentiment > 0.2:
            for ent in nlp(sentence).ents:
                if ent.label_ == "WORK_OF_ART":
                    candidate = normalize_title(ent.text)
                    if candidate and candidate != watched_title and candidate in valid_titles:
                        recommendations.append({
                            "Movie Title": candidate.title(),
                            "Source File": filename.replace(".txt", ""),
                            "Mentions": 1,
                            "Context": sentence,
                            "Full Comment": sentence,
                            "Already Watched": "Yes" if candidate in watched_titles else "No",
                            "Tier": get_sentiment_tier(sentiment),
                            "Sentiment": round(sentiment, 3)
                        })
                    else:
                        reason = "Already Watched" if candidate == watched_title else (
                            "Not in Master List" if candidate not in valid_titles else "Unknown")
                        rejected_titles[candidate]["Count"] += 1
                        rejected_titles[candidate]["Reason"] = reason

# --- Aggregate recommendations ---
counter = Counter((r["Movie Title"], r["Source File"]) for r in recommendations)
aggregated = []
for (movie, source), count in counter.items():
    match = next(r for r in recommendations if r["Movie Title"] == movie and r["Source File"] == source)
    aggregated.append({
        "Movie Title": movie,
        "Source File": source,
        "Mentions": count,
        "Context Example": match["Context"],
        "Full Comment": match["Full Comment"],
        "Already Watched": match["Already Watched"],
        "Tier": match["Tier"],
        "Avg Sentiment": match["Sentiment"]
    })

# --- Save Outputs ---
df_output = pd.DataFrame(aggregated).sort_values(by="Mentions", ascending=False)
df_output.to_csv(OUTPUT_RECOMMENDATIONS, index=False)

summary = df_output["Tier"].value_counts().reset_index()
summary.columns = ["Tier", "Count"]
summary.to_csv(OUTPUT_SUMMARY, index=False)

plt.figure(figsize=(6, 4))
colors = {"Viral": "purple", "Hot": "red", "Trending": "orange", "Cult": "blue", "Other": "gray"}
plt.bar(summary["Tier"], summary["Count"], color=[colors.get(t, "gray") for t in summary["Tier"]])
plt.title("Recommendation Tier Distribution")
plt.xlabel("Tier")
plt.ylabel("Number of Recommendations")
plt.tight_layout()
plt.savefig(OUTPUT_CHART)
plt.close()

rejected_df = pd.DataFrame([
    {"Rejected Title": k, "Count": v["Count"], "Reason": v["Reason"]}
    for k, v in sorted(rejected_titles.items(), key=lambda x: x[1]["Count"], reverse=True)
])
rejected_df.to_csv(OUTPUT_REJECTIONS, index=False)

suggested = rejected_df[(rejected_df["Reason"] == "Not in Master List") & (rejected_df["Count"] >= 3)]
suggested[["Rejected Title", "Count"]].to_csv(OUTPUT_SUGGESTIONS, index=False)

# --- Merge approved additions ---
if os.path.exists(APPROVED_FILE):
    confirm = input("Merge approved additions into the master list? (y/n): ").strip().lower()
    if confirm == 'y':
        approved_df = pd.read_csv(APPROVED_FILE)
        approved_df["Normalized Title"] = approved_df["Rejected Title"].astype(str).apply(normalize_title)
        new_titles = approved_df[~approved_df["Normalized Title"].isin(valid_titles)].copy()

        if not new_titles.empty:
            new_titles["Genre"] = "Suggested"
            new_titles["Year"] = ""
            new_titles["Language"] = ""
            new_titles.rename(columns={"Rejected Title": "Movie Title"}, inplace=True)
            new_titles = new_titles[["Movie Title", "Genre", "Year", "Language"]]
            df_master = pd.concat([
                df_master[["Movie Title", "Genre", "Year", "Language"]],
                new_titles
            ], ignore_index=True).drop_duplicates(subset="Movie Title")
            df_master.to_csv(MASTER_LIST_PATH, index=False)
            print(f"📌 Added {len(new_titles)} new titles to the master list.")
    else:
        print("[INFO] Merge cancelled.")
else:
    print(f"[INFO] No approved additions found: {APPROVED_FILE}")

# --- Final Output ---
print(f"\n✅ Extracted {len(df_output)} entries from {len(comment_files)} files.")
print(f"📄 Output: {OUTPUT_RECOMMENDATIONS}")
print(f"📊 Summary: {OUTPUT_SUMMARY}")
print(f"🖼️ Chart: {OUTPUT_CHART}")
print(f"📎 Rejections: {OUTPUT_REJECTIONS}")
print(f"🧠 Suggestions: {OUTPUT_SUGGESTIONS}")
