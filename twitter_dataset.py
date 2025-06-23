import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from PIL import Image
import os

#1. 데이터 불러오기
df = pd.read_csv("C:/Users/AI-KYJ/Desktop/2025/capstone/dataset/twitter_dataset.csv")  # 경로 확인 필요

#2. 데이터 기본 정보
print(df.head())
print("행 / 열 수:", df.shape)
print(df.info())
print(df.isnull().sum())
print("중복된 행:", df.duplicated().sum())

#3. 전처리
df['Text'] = df['Text'].str.replace(r'http\S+|www.\S+', '', regex=True)  # URL 제거
df['Text'] = df['Text'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)  # 특수문자 제거

#4. 토큰화 / 불용어 제거 / 어간 추출
nltk.download('punkt')
nltk.download('punkt_tab') # 내부적으로 필요한 폴더/파일을 포함하는 구성요소
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

#4-1. 생성되지 못한 토큰열 생성
df['tokens'] = df['Text'].apply(lambda x: nltk.word_tokenize(x) if pd.notnull(x) else [])


# 불용어 제거
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

# 어간 추출
df['tokens'] = df['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])


#5. 감정 분석
df['sentiment_polarity'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['Sentiment Category'] = df['sentiment_polarity'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

#6. 기본 통계
print("평균 리트윗 수:", df['Retweets'].mean())
print("중간 좋아요 수:", df['Likes'].median())
print("리트윗-좋아요 상관계수:", df['Retweets'].corr(df['Likes']))

#7. 히스토그램: 감정 점수
plt.hist(df['sentiment_polarity'], bins=10, range=(-1, 1), edgecolor='black')
plt.title('Sentiment Polarity Distribution')
plt.xlabel('Polarity')
plt.ylabel('Frequency')
plt.show()

#8. 파이차트: 감정 분포
counts = df['Sentiment Category'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Sentiment Category Distribution')
plt.axis('equal')
plt.show()

#9. CountPlot
sns.countplot(x='Sentiment Category', data=df)
plt.title('Count of Tweets by Sentiment Category')
plt.show()

#10. 산점도: 리트윗 vs 좋아요
sns.scatterplot(x='Retweets', y='Likes', data=df)
plt.title('Retweets vs Likes')
plt.show()

#11. 리트윗 분포 히스토그램
sns.histplot(df['Retweets'], kde=True)
plt.title('Retweet Count Distribution')
plt.show()

#12. BoxPlot: 감정별 좋아요
sns.boxplot(x='Sentiment Category', y='Likes', data=df)
plt.title('Likes by Sentiment')
plt.show()

#13. Heatmap: 상관관계
corr = df[['Retweets', 'Likes', 'sentiment_polarity']].corr()
sns.heatmap(corr, annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.show()

#14. 워드클라우드 (기본)
all_text = ' '.join(df['Text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')
plt.show()

#15. 워드클라우드 (마스크 이미지 사용 - 선택)
mask_path = "C:/dataset/mask.jpg"
if os.path.exists(mask_path):
    mask = np.array(Image.open(mask_path))
    wordcloud = WordCloud(width=800, height=400, mask=mask, contour_color='steelblue').generate(all_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Masked Word Cloud')
    plt.show()

#16. 라인플롯: 일별 트윗 수
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
df['Tweet_ID'].resample('D').count().plot(figsize=(12, 6))
plt.title('Daily Tweet Count')
plt.xlabel('Date')
plt.ylabel('Tweet Count')
plt.show()

#17. Barplot: 가장 많이 사용된 단어
words = all_text.split()
word_freq = pd.Series(words).value_counts().head(10)
word_freq.plot(kind='bar', figsize=(10, 6))
plt.title('Top 10 Most Frequent Words')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.show()
