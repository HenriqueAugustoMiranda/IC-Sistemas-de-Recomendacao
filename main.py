import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt


DATASET_PATH = "amazon.csv"


def get_nusers(dataset):

    users = set()

    for user_id in dataset['user_id']:
        users.add(user_id)
 
    return users


def get_nitens(dataset):

    itens = set()

    for user_id in dataset['product_id']:
        itens.add(user_id)
 
    return itens


def get_user_rating_profile(dataset):

    grouped_dataset = dataset.groupby('user_id')
    avg_rating_per_user = grouped_dataset['rating'].mean()

    users_stats = pnd.DataFrame()
    users_stats['user'] = avg_rating_per_user.index
    users_stats['avg_rating'] = avg_rating_per_user.values
    users_stats['variancia_rating'] = grouped_dataset['rating'].var().values

    min_avg_rating = users_stats['avg_rating'].min()
    max_avg_rating = users_stats['avg_rating'].max()

    if max_avg_rating != min_avg_rating:
        users_stats['generoso_ou_critico'] = ((users_stats['avg_rating'] - min_avg_rating) / (max_avg_rating - min_avg_rating)) * 10

    else:
        users_stats['generoso_ou_critico'] = 5

    return users_stats


def get_product_rating_profile(dataset):

    grouped_dataset = dataset.groupby('product_id')
    avg_rating_per_product = grouped_dataset['rating'].mean()

    product_category = (
        dataset[['product_id', 'main_category']]
        .drop_duplicates()
    )

    product_stats = pnd.DataFrame()
    product_stats['product'] = avg_rating_per_product.index
    product_stats['avg_rating'] = avg_rating_per_product.values
    product_stats['variancia_rating'] = grouped_dataset['rating'].var().values

    product_stats = product_stats.merge(
        product_category,
        left_on='product',
        right_on='product_id',
        how='left'
    )

    product_stats.drop(columns=['product_id'], inplace=True)

    min_avg_rating = product_stats['avg_rating'].min()
    max_avg_rating = product_stats['avg_rating'].max()

    if max_avg_rating != min_avg_rating:
        product_stats['qualidade'] = ((product_stats['avg_rating'] - min_avg_rating) / (max_avg_rating - min_avg_rating)) * 10

    else:
        product_stats['qualidade'] = 5

    return product_stats


def dados_analise(dataset):

    users = get_nusers(dataset)
    itens = get_nitens(dataset)

    sparsity = 1 - (len(dataset)/(len(users)*len(itens)))
    print(f"Esparsidade: {sparsity:.4f}")

    user_activity =  dataset['user_id'].value_counts()
    print("\nDistribuição de atividade dos usuários:")  
    print(user_activity.describe())
    
    user_rating_profile = get_user_rating_profile(dataset)
    product_rating_profile = get_product_rating_profile(dataset)

    category_distr = dataset['main_category'].value_counts()
    
    user_activity.plot(kind='hist', bins=50, log=True)
    plt.xlabel('Número de interações')
    plt.ylabel('Frequência (log)')
    plt.show()

    product_popularity = (
        dataset
        .groupby('product_id')['user_id']
        .nunique()
        .sort_values(ascending=False)
    )

    product_names = (
        dataset[['product_id', 'product_name']]
        .drop_duplicates()
    )

    top_products = (
        product_popularity
        .head(60)
        .reset_index()
        .merge(product_names, on='product_id', how='left')
    )


    plt.figure(figsize=(20, 12))

    plt.barh(
        top_products['product_name'],
        top_products['user_id']
    )

    plt.ylabel('Produto')
    plt.xlabel('Número de usuários distintos')
    plt.title('Top 60 produtos mais populares')

    plt.gca().invert_yaxis() 

    plt.tight_layout()
    plt.show()

    
def preprocess_dataset(path):

    dataset = pnd.read_csv(path)

    dataset.columns = dataset.columns.str.strip().str.lower()

    dataset = dataset[['product_id', 'product_name', 'user_id', 'rating', 'rating_count', 'category', 'about_product']]

    dataset['user_id'] = dataset['user_id'].astype(str).str.split(',')
    dataset = dataset.explode('user_id')

    dataset['user_id'] = dataset['user_id'].str.strip()
    dataset['product_id'] = dataset['product_id'].str.strip()

    dataset['rating'] = pnd.to_numeric(dataset['rating'], errors='coerce')

    dataset['rating_count'] = (
        dataset['rating_count']
        .astype(str)
        .str.replace(',', '')
    )
    dataset['rating_count'] = pnd.to_numeric(dataset['rating_count'], errors='coerce')

    dataset = dataset.dropna(subset=['user_id', 'product_id', 'rating'])
    dataset = dataset.drop_duplicates(['user_id', 'product_id'])
    
    dataset['main_category'] = dataset['category'].str.split('|').str[0]

    return dataset


def main():

    dataset = preprocess_dataset(DATASET_PATH)

    dados_analise(dataset)    

main()