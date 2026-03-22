import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def most_popular_recommendation(dataset, user_id, top_k=10, exclude_seen=True):

    popularity = (
        dataset['product_id']
        .value_counts()
        .reset_index()
    )
    popularity.columns = ['product_id', 'score']

    if exclude_seen:
        seen_items = dataset[
            dataset['user_id'] == user_id
        ]['product_id']

        popularity = popularity[
            ~popularity['product_id'].isin(seen_items)
        ]

    recommendations = popularity.head(top_k)

    recommendations = recommendations.merge(
        dataset[['product_id', 'product_name']],
        on='product_id',
        how='left'
    ).drop_duplicates('product_id')

    return recommendations


def build_content_model(dataset):

    dataset = dataset.drop_duplicates('product_id')

    tfidf = TfidfVectorizer(stop_words='english')

    tfidf_matrix = tfidf.fit_transform(dataset['about_product'].fillna(''))

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return dataset, similarity_matrix


def content_based_recommendation(dataset, similarity_matrix, product_id, top_k=5):

    dataset = dataset.reset_index(drop=True)

    indices = pd.Series(dataset.index, index=dataset['product_id'])

    idx = indices[product_id]

    similarity_scores = list(enumerate(similarity_matrix[idx]))

    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similarity_scores = similarity_scores[1:top_k+1]

    product_indices = [i[0] for i in similarity_scores]

    return dataset.iloc[product_indices][['product_id', 'product_name']]


def main():

    final_dataset = pd.read_csv("final_dataset.csv")

    content_data, sim_matrix = build_content_model(final_dataset)

    product_id = content_data['product_id'].iloc[0]

    recs = content_based_recommendation(
        content_data,
        sim_matrix,
        product_id,
        top_k=5
    )

    print("\nContent Based Recommendation:")
    print(recs)
    print("================================================")

    user_id = final_dataset['user_id'].iloc[0]

    recs = most_popular_recommendation(
        final_dataset,
        user_id,
        top_k=5,
        exclude_seen=True
    )

    print("\nMost Popular Recommendation:")
    print(recs[['product_id', 'product_name', 'score']])

main()