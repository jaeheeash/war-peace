import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from wordcloud import WordCloud
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

# Code to Scrape Project Gutenberg
def scrape():
    war_and_peace={}
    page = requests.get("https://www.gutenberg.org/files/2600/2600-h/2600-h.htm#link2HCH0001", headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    chapters = soup.find_all('div', class_='chapter')
    print("True")
    book = ""
    if chapters:
        for chapter in chapters:
            title=chapter.find('h2').get_text(strip=True)
            print(title)
            # if it hits a new book, add book title to the dict
            if title[0] != 'C':
                print('here')
                book=title
                war_and_peace[book] = []
            # if still another chapter, put it in book title
            else:
                content = chapter.find_all('p')
                text_content = "\n".join(p.get_text(strip=True) for p in content)
                war_and_peace[book].append({
                    "chapter": title,
                    "text": text_content
                })
    # After war_and_peace is populated
    with open("war_and_peace.json", "w", encoding="utf-8") as f:
        json.dump(war_and_peace, f, ensure_ascii=False, indent=2)

# Flatten scraped data to standardized csv/json
def flatten():
    with open("war_and_peace.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten the nested structure into a list of records
    flat_data = []
    for book_title, chapters in data.items():
        for chapter in chapters:
            flat_data.append({
                "book": book_title,
                "chapter": chapter["chapter"],
                "text": chapter["text"]
            })

    # df = pd.DataFrame(flat_data)
    # df.to_csv("war_and_peace_flat.csv", index=False, encoding='utf-8')
    with open("war_and_peace_flat.json", "w", encoding="utf-8") as f:
        json.dump(flat_data, f, ensure_ascii=False, indent=2)

# analyze json version
def analyze_json():
    with open("war_and_peace_flat.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    characters = ["Andrew", "Pierre", "Natásha", "Nicholas", "Mary"]
    keywords = ["love", "death", "happiness", "God"]
    window_size = 100  # words before and after

    results = defaultdict(lambda: defaultdict(int))

    for entry in data:
        # now in array
        words = entry['text'].split()
        for i, word in enumerate(words):
            for char in characters:
                if re.fullmatch(char, word.strip(",.?!;:")):
                    window = words[max(0, i - window_size): min(len(words), i + window_size)]
                    for key in keywords:
                        count = sum(1 for w in window if key.lower() == w.lower().strip(",.?!;:"))
                        results[char][key] += count

    for char, keyword_counts in results.items():
        print(f"\n{char}:")
        for kw, count in keyword_counts.items():
            print(f"  {kw}: {count}")

# analyze csv version (used the most)
def analyze_csv(book, wind_size, df):

    characters = ["Andrew", "Pierre", "Natásha", "Mary"]
    keywords = ["love", "death", "happy", "life"]
    window_size = wind_size

    raw_counts = defaultdict(lambda: defaultdict(int))
    mention_counts = defaultdict(int)

    for _, row in df.iterrows():
        words = row["text"].split()
        for i, word in enumerate(words):
            stripped = re.sub(r'[^\w]', '', word)

            for char in characters:
                if stripped == char:
                    mention_counts[char] += 1
                    window = words[max(0, i - window_size): min(len(words), i + window_size)]
                    for kw in keywords:
                        count = sum(
                            1 for w in window
                            if lemmatizer.lemmatize(w.strip(".,!?;:").lower()) == lemmatizer.lemmatize(kw.lower())
                        )
                        raw_counts[char][kw] += count

    
    normalized_counts = {
        char: {
            kw: raw_counts[char][kw] / mention_counts[char] if mention_counts[char] > 0 else 0
            for kw in keywords
        }
        for char in characters
    }

    
    raw_df = pd.DataFrame(raw_counts).T.fillna(0).astype(int)
    raw_df.index.name = "Character"

    norm_df = pd.DataFrame(normalized_counts).T.round(3)
    norm_df.index.name = "Character"

    
    raw_df.to_csv(f"character_keyword_counts_raw_{book}_{window_size}.csv")
    norm_df.to_csv(f"character_keyword_counts_normalized_{book}_{window_size}.csv")

    print("Raw Counts:\n", raw_df)
    print("\nNormalized (per mention):\n", norm_df)
    heatmap(book=book, window_size=wind_size)

def analyze_andrei(book_vol, window_size):
    df = pd.read_csv("w&p_text/wp_volume.csv")
    keywords = ["love", "death", "happy", "life"]
    char = "Andrew"

    raw_counts = defaultdict(lambda: defaultdict(int))
    mention_counts = defaultdict(int)

    for key, _ in book_vol.items():
        book_df=df[df['volume']==key]
        for _, row in book_df.iterrows():
            words = row["text"].split()
            for i, word in enumerate(words):
                stripped = re.sub(r'[^\w]', '', word)
                if stripped == char:
                    mention_counts[key] += 1
                    window = words[max(0, i - window_size): min(len(words), i + window_size)]
                    for kw in keywords:
                        count = sum(
                            1 for w in window
                            if lemmatizer.lemmatize(w.strip(".,!?;:").lower()) == lemmatizer.lemmatize(kw.lower())
                        )
                        raw_counts[key][kw] += count
    
    raw_df = pd.DataFrame(raw_counts).T.fillna(0).astype(int)
    raw_df.index.name = "Character"

    normalized_counts = {
        key: {
            kw: raw_counts[key][kw] / mention_counts[key] if mention_counts[key] > 0 else 0
            for kw in keywords
        }
        for key, _ in book_vol.items()
    }

    norm_df = pd.DataFrame(normalized_counts).T.round(3)
    norm_df.index.name = "Character"

    raw_df.to_csv(f"andrei_keyword_counts_raw_{window_size}.csv")
    norm_df.to_csv(f"andrei_keyword_counts_normalized_{window_size}.csv")

    print("Raw Counts:\n", raw_df)
    print("\nNormalized (per mention):\n", norm_df)
    heatmap_andrei(window_size)

def heatmap_andrei(window_size):
    norm_df = pd.read_csv(f"andrei_keyword_counts_normalized_{window_size}.csv", index_col="Character")

    plt.figure(figsize=(8, 5))
    sns.heatmap(norm_df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5)
    plt.title(f"{"Andrei"}: Keyword Frequency per Book (Normalized per Mention & Window Size {window_size})")
    plt.ylabel("Book")
    plt.xlabel("Keyword")
    plt.tight_layout()
    plt.show()

def heatmap(book, window_size):
    norm_df = pd.read_csv(f"character_keyword_counts_normalized_{book}_{window_size}.csv", index_col="Character")
    
    plt.figure(figsize=(8, 5))
    sns.heatmap(norm_df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5)
    plt.title(f"{book}: Keyword Frequency per Character (Normalized per Mention & Window Size {window_size})")
    plt.ylabel("Character")
    plt.xlabel("Keyword")
    plt.tight_layout()
    plt.show()

def main():
    # Organizing Project Gutenberg version to Oxford Four Books
    true_book = {
        'Book 1': ['BOOK ONE: 1805', "BOOK TWO: 1805", "BOOK THREE: 1805"],
        'Book 2': ['BOOK FOUR: 1806', 'BOOK FIVE: 1806 - 07', 'BOOK SIX: 1808 - 10', 'BOOK SEVEN: 1810 - 11', 'BOOK EIGHT: 1811 - 12'],
        'Book 3': ['BOOK NINE: 1812', 'BOOK TEN: 1812', 'BOOK ELEVEN: 1812'],
        'Book 4': ['BOOK TWELVE: 1812', 'BOOK THIRTEEN: 1812', 'BOOK FOURTEEN: 1812', 'BOOK FIFTEEN: 1812 - 13'],
        # 'Epilogue': ['FIRST EPILOGUE: 1813 - 20', 'SECOND EPILOGUE']
    }

    # Multiple Characters Heatmap
    # for key, _ in true_book.items():
    #     book_df=df[df['volume']==key]
    #     analyze_csv(key, 25, book_df)

    # Andrei Heatmap
    analyze_andrei(true_book, 25)
    

if __name__=="__main__":
    main()