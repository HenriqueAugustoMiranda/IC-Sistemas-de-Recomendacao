import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import analise_utils as au

def analise_geral(dataset):

    users = au.get_nusers(dataset)
    itens = au.get_nitens(dataset)

    print("=== Visão geral do dataset ===")
    print(f"Número de usuários: {len(users)}")
    print(f"Número de itens: {len(itens)}")

    sparsity = 1 - (len(dataset) / (len(users) * len(itens)))
    print(f"Esparsidade: {sparsity:.4f}")
    print()


def analise_usuarios(dataset):

    user_rating_profile = au.get_user_rating_profile(dataset)

    plt.figure(figsize=(10, 6))
    plt.hist(user_rating_profile['avg_rating'], bins=40)

    plt.xlabel('Média das avaliações do usuário')
    plt.ylabel('Número de usuários')
    plt.title('Distribuição da média de avaliações dos usuários')

    plt.tight_layout()
    plt.show()


def analise_itens(dataset):

    product_rating_profile = au.get_product_rating_profile(dataset)

    product_popularity = (
        dataset
        .groupby('product_id')['user_id']
        .nunique()
        .reset_index(name='popularidade')
    )

    product_quality_popularity = product_rating_profile.merge(
        product_popularity,
        left_on='product',
        right_on='product_id'
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(
        product_quality_popularity['popularidade'],
        product_quality_popularity['qualidade'],
        alpha=0.4
    )

    plt.xscale('log')
    plt.xlabel('Popularidade (usuários distintos)')
    plt.ylabel('Qualidade normalizada')
    plt.title('Popularidade vs Qualidade dos produtos')

    plt.tight_layout()
    plt.show()

    category_distr = dataset['main_category'].value_counts()

    plt.figure(figsize=(10, 6))
    category_distr.head(15).plot(kind='bar')

    plt.xlabel('Categoria')
    plt.ylabel('Número de interações')
    plt.title('Categorias mais avaliadas')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def analise_interacoes(dataset):

    user_activity = dataset['user_id'].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    plt.hist(user_activity.values, bins=50, log=True)

    plt.xlabel('Número de itens avaliados')
    plt.ylabel('Número de usuários (log)')
    plt.title('Histórico de interações dos usuários')

    plt.tight_layout()
    plt.show()

    sorted_activity = np.sort(user_activity.values)
    cdf = np.arange(1, len(sorted_activity) + 1) / len(sorted_activity)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_activity, cdf)

    plt.xscale('log')
    plt.xlabel('Número de avaliações')
    plt.ylabel('Proporção acumulada de usuários')
    plt.title('Distribuição acumulada de usuários por atividade')

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