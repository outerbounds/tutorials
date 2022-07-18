import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('reviews.csv')
df.columns = ['review_id', 'clothing_id', 'age', 'title', 'review_text', 'rating', 'recommend', 'pos_count', 'div_name', 'dept', 'class']
df = df[~df.review_text.isna()]
df['labels'] = df['rating'] >= 4
df['labels'] = df['labels'].astype(int)

def concat_title(x):
    title = str(x.title) + ': ' if x.title else ''
    review = x.review_text  if x.review_text else ''
    return title + review

df['review'] = df.apply(concat_title, axis=1)
df = df.drop(['review_id', 'clothing_id', 'age', 'recommend', 'pos_count', 'rating', 'title', 'review_text', 'dept', 'class', 'div_name'], axis=1)
df = df.dropna()

pred,train = train_test_split(df, train_size=.1)
train.to_parquet('train.parquet', index=False)
pred.to_parquet('predict.parquet', index=False)



