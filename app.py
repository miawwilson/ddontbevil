from flask import Flask, render_template, request, redirect, url_for
from nltk.corpus import stopwords
import collections
from serpapi import GoogleSearch
from newspaper import Article
from flask_caching import Cache
from flask_redis import FlaskRedis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import nltk
import string
import random
import re


app = Flask(__name__)
app.config["REDIS_URL"] = "rediss://:p9e63812005251c9236ad4c48f31867f9d1802b2963cde6cfd595b9032b4e5ae5@ec2-52-23-144-238.compute-1.amazonaws.com:28850"
redis_client = FlaskRedis(app)
app.jinja_env.globals.update(zip=zip)


@app.route('/')
def index():
    cached_data = redis_client.get('index_data')
    if cached_data:
        return cached_data.decode()
    # Scrape Google News for first 50 search results of 'AI'
    query = 'AI'
    params = {
        'engine': 'google',
        'q': query,
        'tbm': 'nws',
        'num': 100,
        'api_key': '189872377b6b631f9f0e925c3dabe549f96460fe7932b75304a0066e4fc8bd6c',
        'gl': 'uk'
    }
    search = GoogleSearch(params)
    response = search.get_dict()

    if 'news_results' in response:
        news_results = response['news_results']
    else:
        news_results = []

    # Initialize Vader sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    text = ''
    sentiment_scores = []
    positive_sentences = []
    negative_sentences = []
    neutral_sentences = []

    for result in news_results:
        url = result['link']
        article = Article(url)
        try:
            article.download()
            article.parse()
            article_text = article.text
            if not article_text:
                continue

            text += article.title + ' ' + article_text

            # find quotes in the article
            quotes = re.findall(r'"(.*?)"', article_text)

            for quote in quotes:
                sentiment = analyzer.polarity_scores(quote)
                sentiment_score = sentiment['compound']
                print(f"Sentiment score: {sentiment_score}")

                # Depending on the sentiment score, append the quote to the appropriate list
                if sentiment_score > 0.05:
                    positive_sentences.append((quote, url))
                elif sentiment_score < -0.05:
                    negative_sentences.append((quote, url))
                else:
                    neutral_sentences.append((quote, url))

            # Calculate sentiment of full article
            sentiment = analyzer.polarity_scores(article_text)
            sentiment_scores.append(sentiment['compound'])
        except Exception as e:
            print(f"Error processing article: {url} - {e}")

    # Stop words and my own stopwords
    my_stop_words = ['ai', 'artificial', 'intelligence', '-', '_', 'I', '60', ' ',
                     'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                     'saturday', 'sunday', 'like', 'I', '-', 'a.i.', 'and', 'but',
                     'also', 'its', 'said', '--', '—', 'ai,', 'one', 'two', "it's",
                     'said.', "-", "—", "it's", 'way', ]

    stop_words = set(stopwords.words('english')).union(set(my_stop_words))

    # Remove stop words
    words = [word.lower() for word in text.split() if word.lower() not in stop_words]

    # Count word frequency
    word_counts = collections.Counter(words)
    top_words = word_counts.most_common(30)

    # AI Word of the Day
    ai_word_of_the_day = top_words[0][0] if top_words else None  # select the most frequent word

    # Convert top words to a list of (word, count) tuples
    word_list = []
    for word, count in top_words:
        word_list.append((word, count))

    # Calculate average sentiment score
    avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    # Determine mood based on overall sentiment score
    if avg_sentiment_score > 0.05:
        mood = 'Positive'
    elif avg_sentiment_score < -0.05:
        mood = 'Negative'
    else:
        mood = 'Neutral'

    positive_sentiment = sum(1 for score in sentiment_scores if score > 0.1)
    negative_sentiment = sum(1 for score in sentiment_scores if score < -0.1)
    neutral_sentiment = len(sentiment_scores) - positive_sentiment - negative_sentiment

    positive_percentage = (positive_sentiment / len(sentiment_scores)) * 100 if sentiment_scores else 0
    negative_percentage = (negative_sentiment / len(sentiment_scores)) * 100 if sentiment_scores else 0
    neutral_percentage = (neutral_sentiment / len(sentiment_scores)) * 100 if sentiment_scores else 0

    random_positive_sentence, random_positive_sentence_url = random.choice(
        positive_sentences) if positive_sentences else (None, None)
    random_negative_sentence, random_negative_sentence_url = random.choice(
        negative_sentences) if negative_sentences else (None, None)
    random_neutral_sentence, random_neutral_sentence_url = random.choice(
        neutral_sentences) if neutral_sentences else (None, None)

    print("Positive Sentences: ", random_positive_sentence)
    print("Negative Sentences: ", random_negative_sentence)
    print("Neutral Sentences: ", random_neutral_sentence)

    data = render_template('index.html', word_list=word_list, mood=mood, avg_sentiment_score=avg_sentiment_score,
                           sentiment_data=[positive_percentage, negative_percentage, neutral_percentage],
                           ai_word_of_the_day=ai_word_of_the_day, random_positive_sentence=random_positive_sentence,
                           random_positive_sentence_url=random_positive_sentence_url,
                           random_negative_sentence=random_negative_sentence,
                           random_negative_sentence_url=random_negative_sentence_url,
                           random_neutral_sentence=random_neutral_sentence,
                           random_neutral_sentence_url=random_neutral_sentence_url)

    redis_client.set('index_data', data)

    return data


@app.route('/articles/<word>')
def articles(word):
    cache_key = f'articles_data_{word}'
    cached_data = redis_client.get('cache_key')
    if cached_data:
        return cached_data.decode()
    query = f'{word} AI'
    params = {
        'engine': 'google',
        'q': query,
        'tbm': 'nws',
        'num': 100,
        'api_key': '189872377b6b631f9f0e925c3dabe549f96460fe7932b75304a0066e4fc8bd6c'
    }
    search = GoogleSearch(params)
    news_results = search.get_dict()

# Extract articles containing word
    articles = []
    for result in news_results['news_results']:
        if word.lower() in result['title'].lower() or word.lower() in result['snippet'].lower():
            article = {'title': result['title'], 'snippet': result['snippet'], 'url': result['link']}
            articles.append(article)

    if not articles:
        return render_template('articles.html', word=word)

    data = render_template('articles.html', word=word, articles=articles,)
    redis_client.set(cache_key, data)
    redis_client.expire(cache_key, 3600)
    return data


@app.route('/companies/')
def companies():
    cached_data = redis_client.get('companies_data')
    if cached_data:
        return cached_data.decode()
    company_list = ['Google', 'OpenAI', 'Amazon', 'Microsoft']
    all_articles = []
    all_word_lists = []
    all_sentiment_scores = []
    sentiment_scores = []
    analyzer = SentimentIntensityAnalyzer()

    for company in company_list:
        text = ''
        sentiment_scores = []
        query = f'{company} AI'
        params = {
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 50,
            'api_key': '189872377b6b631f9f0e925c3dabe549f96460fe7932b75304a0066e4fc8bd6c',
            'gl': 'uk'
        }
        search = GoogleSearch(params)
        news_results = search.get_dict()
        company_sentiment_scores = []
        company_articles = []

        for result in news_results['news_results']:
            article = {
                'title': result['title'],
                'snippet': result['snippet'],
                'url': result['link']
            }
            company_articles.append(article)

            # Download full article
            url = result['link']
            article = Article(url)
            try:
                article.download()
                article.parse()
                article_text = article.text
                if not article_text:
                    continue

                text += article.title + ' ' + article_text

                # Calculate sentiment of full article
                sentiment = analyzer.polarity_scores(article_text)
                sentiment_scores.append(sentiment['compound'])
            except Exception as e:
                print(f"Error processing article: {url} - {e}")

            # Calculate average sentiment score for the company
        avg_company_sentiment_score = sum(company_sentiment_scores) / len(
            company_sentiment_scores) if company_sentiment_scores else 0
        all_sentiment_scores.append({company: avg_company_sentiment_score})

        all_articles.append({company: company_articles})

        # Stop words and my own stopwords
        my_stop_words = ['ai', 'artificial', 'intelligence', '-', '_', 'I', '60', ' ',
                         'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                         'saturday', 'sunday', 'like', 'I', '-', 'a.i.', 'and', 'but',
                         'also', 'its', 'said', '--', '—', 'ai,', 'one', 'two', "it's",
                         'said.', "-", "—", "it's", 'way', 'openai', 'google', 'microsoft', 'amazon', "google's"]

        stop_words = set(stopwords.words('english')).union(set(my_stop_words))

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove stop words
        words = [word.lower() for word in text.split() if word.lower() not in stop_words]

        # Count word frequency
        word_counts = collections.Counter(words)
        top_words = word_counts.most_common(30)

        # Convert top words to a list of (word, count) tuples
        word_list = []
        for word, count in top_words:
            word_list.append((word, count))

        all_word_lists.append({company: word_list})
    # Calculate average sentiment score
    avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    # Determine mood based on overall sentiment score
    if avg_sentiment_score > 0.05:
        mood = 'Positive'
    elif avg_sentiment_score < -0.05:
        mood = 'Negative'
    else:
        mood = 'Neutral'

    positive_sentiment = sum(1 for score in sentiment_scores if score > 0.05)
    negative_sentiment = sum(1 for score in sentiment_scores if score < -0.05)
    neutral_sentiment = len(sentiment_scores) - positive_sentiment - negative_sentiment

    positive_percentage = (positive_sentiment / len(sentiment_scores)) * 100 if sentiment_scores else 0
    negative_percentage = (negative_sentiment / len(sentiment_scores)) * 100 if sentiment_scores else 0
    neutral_percentage = (neutral_sentiment / len(sentiment_scores)) * 100 if sentiment_scores else 0

    data = render_template('companies.html', all_articles=all_articles, all_word_lists=all_word_lists, mood=mood,
                           avg_sentiment_score=avg_sentiment_score, sentiment_data=[positive_percentage,
                           negative_percentage, neutral_percentage], all_sentiment_scores=all_sentiment_scores)

    redis_client.set('companies_data', data)
    redis_client.expire('companies_data', 3600)
    return data


@app.route('/wordcloud/')
def wordcloud():
    cached_data = redis_client.get('wordcloud_data')
    if cached_data:
        return cached_data.decode()

    # Scrape Google News for first 50 search results of 'AI'
    query = 'AI'
    params = {
        'engine': 'google',
        'q': query,
        'tbm': 'nws',
        'num': 100,
        'api_key': '189872377b6b631f9f0e925c3dabe549f96460fe7932b75304a0066e4fc8bd6c',
        'gl': 'uk'
    }
    search = GoogleSearch(params)
    response = search.get_dict()

    if 'news_results' in response:
        news_results = response['news_results']
    else:
        news_results = []

    # Initialize Vader sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    text = ''
    sentiment_scores = []
    for result in news_results:
        url = result['link']
        article = Article(url)
        try:
            article.download()
            article.parse()
            article_text = article.text
            if not article_text:
                continue

            text += article.title + ' ' + article_text

            # Calculate sentiment of full article
            sentiment = analyzer.polarity_scores(article_text)
            sentiment_scores.append(sentiment['compound'])
        except Exception as e:
            print(f"Error processing article: {url} - {e}")

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Stop words and my own stopwords
    my_stop_words = ['ai', 'artificial', 'intelligence', '-', '_', 'I', '60', ' ',
                     'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                     'saturday', 'sunday', 'like', 'I', '-', 'a.i.', 'and', 'but',
                     'also', 'its', 'said', '--', '—', 'ai,', 'one', 'two', "it's",
                     'said.', "-", "—", "it's", 'way', ]

    stop_words = set(stopwords.words('english')).union(set(my_stop_words))

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stop words
    words = [word.lower() for word in text.split() if word.lower() not in stop_words]

    # Count word frequency
    word_counts = collections.Counter(words)
    top_words = word_counts.most_common(30)

    # Convert top words to a list of (word, count) tuples
    word_list = []
    for word, count in top_words:
        word_list.append((word, count))

    data = render_template('wordcloud.html', word_list=word_list)

    redis_client.set('wordcloud_data', data)

    return data


@app.route('/mood/')
def mood():
    cached_data = redis_client.get('mood_data')
    if cached_data:
        return cached_data.decode()

    query = 'AI'
    params = {
        'engine': 'google',
        'q': query,
        'tbm': 'nws',
        'num': 100,
        'api_key': '189872377b6b631f9f0e925c3dabe549f96460fe7932b75304a0066e4fc8bd6c',
        'gl': 'uk'
    }
    search = GoogleSearch(params)
    response = search.get_dict()

    news_results = response.get('news_results', [])

    positive_sentences = []
    negative_sentences = []
    neutral_sentences = []

    sid = SentimentIntensityAnalyzer()

    for result in news_results:
        url = result['link']
        article = Article(url)
        try:
            article.download()
            article.parse()
            article_text = article.text
            if not article_text:
                continue

            # Split the article into sentences
            sentences = article_text.split('.')
            for sentence in sentences:
                ss = sid.polarity_scores(sentence)
                if ss['compound'] >= 0.05:
                    positive_sentences.append(sentence)
                elif ss['compound'] <= -0.05:
                    negative_sentences.append(sentence)
                else:
                    neutral_sentences.append(sentence)

        except Exception as e:
            print(f"Error processing article: {url} - {e}")

    # Remove punctuation
    positive_sentences = [sentence.translate(str.maketrans('', '', string.punctuation))
                          for sentence in positive_sentences]
    negative_sentences = [sentence.translate(str.maketrans('', '', string.punctuation))
                          for sentence in negative_sentences]
    neutral_sentences = [sentence.translate(str.maketrans('', '', string.punctuation))
                         for sentence in neutral_sentences]

    # Stop words and my own stopwords
    my_stop_words = ['ai', 'artificial', 'intelligence', '-', '_', 'I', '60', ' ',
                     'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                     'saturday', 'sunday', 'like', 'I', '-', 'a.i.', 'and', 'but',
                     'also', 'its', 'said', '--', '—', 'ai,', 'one', 'two', "it's",
                     'said.', "-", "—", "it's", 'way', '"', ]
    # your remaining code
    stop_words = set(stopwords.words('english')).union(set(my_stop_words))

    # Extract words from sentences and remove stop words
    positive_words = [word.lower() for sentence in positive_sentences for word in sentence.split() if
                      word.lower() not in stop_words]
    neutral_words = [word.lower() for sentence in neutral_sentences for word in sentence.split() if
                     word.lower() not in stop_words]
    negative_words = [word.lower() for sentence in negative_sentences for word in sentence.split() if
                      word.lower() not in stop_words]

    # Count word frequency
    positive_word_counts = collections.Counter(positive_words)
    neutral_word_counts = collections.Counter(neutral_words)
    negative_word_counts = collections.Counter(negative_words)

    # Get all unique words
    all_words = set(positive_words + neutral_words + negative_words)

    # For each unique word, count the number of lists it appears in
    word_list_counts = {
        word: (word in positive_word_counts) + (word in neutral_word_counts) + (word in negative_word_counts) for word
        in all_words}

    # Only keep words that appear in one list
    unique_words = {word for word, count in word_list_counts.items() if count == 1}

    # Filter each word count list to only include unique words
    positive_word_counts = {word: count for word, count in positive_word_counts.items() if word in unique_words}
    neutral_word_counts = {word: count for word, count in neutral_word_counts.items() if word in unique_words}
    negative_word_counts = {word: count for word, count in negative_word_counts.items() if word in unique_words}

    # Get the top 30 words from each list
    top_positive_words = sorted(positive_word_counts.items(), key=lambda item: item[1], reverse=True)[:30]
    top_neutral_words = sorted(neutral_word_counts.items(), key=lambda item: item[1], reverse=True)[:30]
    top_negative_words = sorted(negative_word_counts.items(), key=lambda item: item[1], reverse=True)[:30]

    # Convert top words to a list of (word, count) tuples
    positive_word_list = [(word, count) for word, count in top_positive_words]
    neutral_word_list = [(word, count) for word, count in top_neutral_words]
    negative_word_list = [(word, count) for word, count in top_negative_words]

    positive_count = len(positive_sentences)
    neutral_count = len(neutral_sentences)
    negative_count = len(negative_sentences)

    # Determine prevalent mood
    if positive_count > negative_count and positive_count > neutral_count:
        prevalent_mood = 'Positive'
    elif negative_count > positive_count and negative_count > neutral_count:
        prevalent_mood = 'Negative'
    else:
        prevalent_mood = 'Neutral'

    data = render_template('mood.html',
                           positive_word_list=positive_word_list,
                           neutral_word_list=neutral_word_list,
                           negative_word_list=negative_word_list,
                           positive_sentences=positive_sentences,
                           negative_sentences=negative_sentences,
                           neutral_sentences=neutral_sentences,
                           positive_count=len(positive_sentences),
                           neutral_count=len(neutral_sentences),
                           negative_count=len(negative_sentences),
                           prevalent_mood=prevalent_mood)

    redis_client.set('mood_data', data)

    return data


@app.route('/userinput/', methods=['GET', 'POST'])
def userinput():
    if request.method == 'POST':
        user_query = request.form.get('user_query')

        # concatenate user input with search term "AI"
        query = 'AI ' + user_query

        params = {
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'api_key': '189872377b6b631f9f0e925c3dabe549f96460fe7932b75304a0066e4fc8bd6c',
            'gl': 'uk'
        }
        search = GoogleSearch(params)
        response = search.get_dict()

        if 'news_results' in response:
            news_results = response['news_results']
        else:
            news_results = []

        # Initialize Vader sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()

        text = ''
        sentiment_scores = []
        for result in news_results:
            url = result['link']
            article = Article(url)
            try:
                article.download()
                article.parse()
                article_text = article.text
                if not article_text:
                    continue

                text += article.title + ' ' + article_text

                # Calculate sentiment of full article
                sentiment = analyzer.polarity_scores(article_text)
                sentiment_scores.append(sentiment['compound'])
            except Exception as e:
                print(f"Error processing article: {url} - {e}")

        # Stop words and my own stopwords
        my_stop_words = ['ai', 'artificial', 'intelligence', user_query, '-', '_', 'I', '60', ' ',
                         'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                         'saturday', 'sunday', 'like', 'I', '-', 'a.i.', 'and', 'but',
                         'also', 'its', 'said', '--', '—', 'ai,', 'one', 'two', "it's",
                         'said.', "-", "—", "it's", 'way', ]

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        stop_words = set(stopwords.words('english')).union(set(my_stop_words))

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove stop words
        words = [word.lower() for word in text.split() if word.lower() not in stop_words]

        # Count word frequency
        word_counts = collections.Counter(words)
        top_words = word_counts.most_common(30)

        # Convert top words to a list of (word, count) tuples
        word_list = []
        for word, count in top_words:
            word_list.append((word, count))

        data = render_template('userinput.html', word_list=word_list)

        redis_client.set('userinput_data', data)

        return data
    else:
        # Handle the GET request
        return render_template('userinput.html')


if __name__ == '__main__':
    app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    
