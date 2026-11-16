
import streamlit as st
import math
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Search Engine",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------------
# DOCUMENT COLLECTION
# -----------------------------------

documents = {
    "Google Search Technology": """
        Google is a search engine that helps users find
        information on the internet. Search engines crawl
        websites, index information and rank search results.
    """,

    "How Search Engines Work": """
        Search engines work through crawling, indexing
        and ranking. Crawlers discover web pages and
        search indexes store information for retrieval.
    """,

    "Internet and Websites": """
        The internet connects millions of websites.
        Websites contain text, images, videos and
        information that users can search online.
    """,

    "Natural Language Processing": """
        Natural language processing helps computers
        understand human language. It uses tokenization,
        text processing and machine learning.
    """,

    "Information Retrieval": """
        Information retrieval finds relevant documents
        based on a user's search query. TF-IDF and
        cosine similarity help rank documents.
    """,

    "Python Programming": """
        Python is a programming language used in
        data science, artificial intelligence,
        machine learning and search engine development.
    """,

    "Web Crawling": """
        Web crawlers automatically visit websites,
        discover links and collect web page information
        for search engine indexing.
    """,

    "Search Engine Ranking": """
        Search engines rank web pages according to
        relevance and other ranking factors.
        Relevant results appear higher in search results.
    """
}

# -----------------------------------
# TEXT PREPROCESSING
# -----------------------------------

stop_words = {
    "the", "is", "are", "a", "an", "and", "or",
    "in", "on", "of", "to", "for", "that", "with",
    "it", "by", "as", "this", "be", "from", "how"
}

def preprocess(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return [
        word for word in words
        if word not in stop_words
    ]

# -----------------------------------
# CREATE INVERTED INDEX
# -----------------------------------

inverted_index = defaultdict(list)

for title, content in documents.items():
    words = set(preprocess(content))
    for word in words:
        inverted_index[word].append(title)

# -----------------------------------
# TF-IDF CALCULATION
# -----------------------------------

processed_docs = {
    title: preprocess(content)
    for title, content in documents.items()
}

N = len(processed_docs)

document_frequency = Counter()

for words in processed_docs.values():
    document_frequency.update(set(words))

def calculate_tfidf(words):
    frequency = Counter(words)
    scores = {}

    for word, count in frequency.items():
        tf = count / len(words)
        idf = math.log((N + 1) / (document_frequency[word] + 1)) + 1
        scores[word] = tf * idf

    return scores

tfidf_vectors = {
    title: calculate_tfidf(words)
    for title, words in processed_docs.items()
}

# -----------------------------------
# COSINE SIMILARITY
# -----------------------------------

def cosine_similarity(vector1, vector2):

    common_words = set(vector1) & set(vector2)

    dot_product = sum(
        vector1[word] * vector2[word]
        for word in common_words
    )

    magnitude1 = math.sqrt(
        sum(value ** 2 for value in vector1.values())
    )

    magnitude2 = math.sqrt(
        sum(value ** 2 for value in vector2.values())
    )

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)

# -----------------------------------
# SEARCH FUNCTION
# -----------------------------------

def search_engine(query):

    query_words = preprocess(query)

    if not query_words:
        return []

    query_vector = calculate_tfidf(query_words)

    results = []

    for title, document_vector in tfidf_vectors.items():

        score = cosine_similarity(
            query_vector,
            document_vector
        )

        if score > 0:
            results.append((title, score))

    results.sort(key=lambda x: x[1], reverse=True)

    return results

# -----------------------------------
# WEBSITE DESIGN
# -----------------------------------

st.markdown(
    """
    <h1 style='text-align:center; color:#2563EB;'>
    🔍 SEARCH ENGINE
    </h1>
    <h3 style='text-align:center;'>
    How Google Finds Information
    </h3>
    <p style='text-align:center; color:gray;'>
    Information Retrieval System using Python
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

st.subheader("Explore Information")

query = st.text_input(
    "Enter your search query",
    placeholder="Example: How do search engines work?"
)

search_button = st.button(
    "🔍 Search",
    type="primary",
    use_container_width=True
)

if search_button:

    if query.strip():

        results = search_engine(query)

        st.subheader("Search Results")

        if results:

            st.success(
                f"Found {len(results)} relevant documents."
            )

            for rank, (title, score) in enumerate(results, 1):

                with st.container(border=True):

                    st.markdown(f"### {rank}. {title}")

                    st.write(documents[title].strip())

                    st.progress(min(score, 1.0))

                    st.caption(
                        f"Cosine Similarity: {score:.4f}"
                    )

            # -----------------------------------
            # RESULT VISUALIZATION
            # -----------------------------------

            st.subheader("Search Result Ranking")

            titles = [item[0] for item in results]
            scores = [item[1] for item in results]

            fig, ax = plt.subplots(figsize=(10, 4))

            ax.barh(titles[::-1], scores[::-1])

            ax.set_xlabel("Cosine Similarity Score")
            ax.set_title("Document Relevance Ranking")

            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        else:
            st.warning(
                "No relevant documents found. Try another query."
            )

    else:
        st.warning("Please enter a search query.")

# -----------------------------------
# SIDEBAR
# -----------------------------------

with st.sidebar:

    st.title("About Project")

    st.write(
        "This project demonstrates how a basic "
        "search engine retrieves and ranks information."
    )

    st.divider()

    st.subheader("Technologies")

    st.write("🐍 Python")
    st.write("📓 Jupyter Notebook")
    st.write("⚙️ TF-IDF")
    st.write("📊 Cosine Similarity")
    st.write("🔍 Inverted Index")

    st.divider()

    st.write("Total Documents:", len(documents))
    st.write("Indexed Words:", len(inverted_index))

    st.caption("Academic Information Retrieval Project")

