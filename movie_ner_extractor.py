# movie_recommendation_extractor.py

import os
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm
from textblob import TextBlob

# --- Configuration ---
CONFIG = {
    "MASTER_LIST_PATH": "./tmdb_master_list.csv",
    "COMMENTS_FOLDER": "./comments",
    "OUTPUT_RECOMMENDATIONS": "ner_movie_recommendations.csv",
    "OUTPUT_SUMMARY": "ner_summary_by_tier.csv",
    "OUTPUT_REJECTIONS": "rejected_titles_log.csv",
    "OUTPUT_SUGGESTIONS": "suggested_master_list_additions.csv",
    "OUTPUT_CHART": "tier_distribution_chart.png",
    "APPROVED_FILE": "approved_master_additions.csv",
    "BLACKLIST_FILE": "blacklist.txt",
    "AUTO_APPROVE": True,
    "MIN_MENTIONS_FOR_APPROVAL": 3,
    "ALLOWED_TIERS_FOR_APPROVAL": ["Viral", "Hot"]
}

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

def load_master_list(path):
    df = pd.read_csv(path)
    df["Normalized Title"] = df["Movie Title"].astype(str).apply(normalize_title)
    return df, set(df["Normalized Title"])

def load_blacklist(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return {normalize_title(line) for line in f if line.strip()}
    return set()

def extract_recommendations(comments_folder, valid_titles, watched_titles, blacklist):
    recommendations = []
    rejected_titles = defaultdict(lambda: {"Count": 0, "Reason": ""})
    comment_files = [f for f in os.listdir(comments_folder) if f.endswith(".txt")]

    for filename in tqdm(comment_files, desc="Processing comments", unit="file"):
        watched_title = normalize_title(filename)
        with open(os.path.join(comments_folder, filename), "r", encoding="utf-8") as file:
            text = file.read()

        doc = nlp(text)
        for sent in doc.sents:
            sentence = sent.text.strip()
            if len(sentence.split()) < 3:
                continue
            sentiment = TextBlob(sentence).sentiment.polarity
            if sentiment <= 0.2:
                continue

            for ent in sent.ents:
                if ent.label_ == "WORK_OF_ART":
                    candidate = normalize_title(ent.text)
                    if not candidate or candidate == watched_title:
                        continue
                    if candidate in blacklist:
                        reason = "Blacklisted"
                    elif candidate not in valid_titles:
                        reason = "Not in Master List"
                    else:
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
                        continue

                    rejected_titles[candidate]["Count"] += 1
                    rejected_titles[candidate]["Reason"] = reason

    return recommendations, rejected_titles

def aggregate_recommendations(recommendations):
    counter = Counter((r["Movie Title"], r["Source File"]) for r in recommendations)
    lookup = {(r["Movie Title"], r["Source File"]): r for r in recommendations}

    aggregated = []
    for (movie, source), count in counter.items():
        match = lookup[(movie, source)]
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
    return pd.DataFrame(aggregated)

def save_outputs(df_output, rejected_titles):
    df_output.to_csv(CONFIG["OUTPUT_RECOMMENDATIONS"], index=False)

    summary = df_output["Tier"].value_counts().reset_index()
    summary.columns = ["Tier", "Count"]
    summary.to_csv(CONFIG["OUTPUT_SUMMARY"], index=False)

    plt.figure(figsize=(6, 4))
    colors = {"Viral": "purple", "Hot": "red", "Trending": "orange", "Cult": "blue", "Other": "gray"}
    plt.bar(summary["Tier"], summary["Count"], color=[colors.get(t, "gray") for t in summary["Tier"]])
    plt.title("Recommendation Tier Distribution")
    plt.xlabel("Tier")
    plt.ylabel("Number of Recommendations")
    plt.tight_layout()
    plt.savefig(CONFIG["OUTPUT_CHART"])
    plt.close()

    rejected_df = pd.DataFrame([
        {"Rejected Title": k, "Count": v["Count"], "Reason": v["Reason"]}
        for k, v in sorted(rejected_titles.items(), key=lambda x: x[1]["Count"], reverse=True)
    ])
    rejected_df.to_csv(CONFIG["OUTPUT_REJECTIONS"], index=False)

    suggested = rejected_df[(rejected_df["Reason"] == "Not in Master List") & (rejected_df["Count"] >= CONFIG["MIN_MENTIONS_FOR_APPROVAL"])]
    suggested[["Rejected Title", "Count"]].to_csv(CONFIG["OUTPUT_SUGGESTIONS"], index=False)

    if CONFIG["AUTO_APPROVE"] and not suggested.empty:
        approved_df = suggested.copy()
        approved_df["Genre"] = "Suggested"
        approved_df["Year"] = ""
        approved_df["Language"] = ""
        approved_df.to_csv(CONFIG["APPROVED_FILE"], index=False)
        print(f"✅ Auto-approved {len(approved_df)} titles based on mention count.")

def merge_approved_additions(df_master, valid_titles):
    approved_path = CONFIG["APPROVED_FILE"]
    if not os.path.exists(approved_path):
        print(f"[INFO] No approved additions found: {approved_path}")
        return df_master

    confirm = 'y' if CONFIG["AUTO_APPROVE"] else input("Merge approved additions into the master list? (y/n): ").strip().lower()
    if confirm != 'y':
        print("[INFO] Merge cancelled.")
        return df_master

    approved_df = pd.read_csv(approved_path)
    approved_df["Normalized Title"] = approved_df["Rejected Title"].astype(str).apply(normalize_title)
    new_titles = approved_df[~approved_df["Normalized Title"].isin(valid_titles)].copy()

    if not new_titles.empty:
        new_titles.rename(columns={"Rejected Title": "Movie Title"}, inplace=True)
        new_titles = new_titles[["Movie Title", "Genre", "Year", "Language"]]
        df_master = pd.concat([
            df_master[["Movie Title", "Genre", "Year", "Language"]],
            new_titles
        ], ignore_index=True).drop_duplicates(subset="Movie Title")
        df_master.to_csv(CONFIG["MASTER_LIST_PATH"], index=False)
        print(f"📌 Added {len(new_titles)} new titles to the master list.")

    return df_master

def main():
    df_master, valid_titles = load_master_list(CONFIG["MASTER_LIST_PATH"])
    blacklist = load_blacklist(CONFIG["BLACKLIST_FILE"])
    comment_files = [f for f in os.listdir(CONFIG["COMMENTS_FOLDER"]) if f.endswith(".txt")]
    watched_titles = {normalize_title(f) for f in comment_files}

    print("✅ Loaded and normalized TMDb master list")
    print(f"🔄 Starting processing of {len(comment_files)} files...")

    recommendations, rejected_titles = extract_recommendations(
        CONFIG["COMMENTS_FOLDER"], valid_titles, watched_titles, blacklist)

    df_output = aggregate_recommendations(recommendations)
    save_outputs(df_output, rejected_titles)

    df_master = merge_approved_additions(df_master, valid_titles)

    print(f"\n✅ Extracted {len(df_output)} entries from {len(comment_files)} files.")
    print(f"📄 Output: {CONFIG['OUTPUT_RECOMMENDATIONS']}")
    print(f"📊 Summary: {CONFIG['OUTPUT_SUMMARY']}")
    print(f"🖼️ Chart: {CONFIG['OUTPUT_CHART']}")
    print(f"📎 Rejections: {CONFIG['OUTPUT_REJECTIONS']}")
    print(f"🧠 Suggestions: {CONFIG['OUTPUT_SUGGESTIONS']}")

if __name__ == "__main__":
    main()
