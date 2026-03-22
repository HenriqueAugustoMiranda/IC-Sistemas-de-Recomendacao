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

    counts = user_rating_profile['count_ratings']

    p1 = counts.quantile(0.33)
    p2 = counts.quantile(0.66)

    user_rating_profile['nivel_atividade'] = np.select(
        [
            counts <= p1,
            (counts > p1) & (counts <= p2),
            counts > p2
        ],
        [
            'pouco ativo',
            'medio ativo',
            'muito ativo'
        ],
        default='indefinido'
    )

    activity_counts = user_rating_profile['nivel_atividade'].value_counts()

    plt.figure(figsize=(8, 6))
    activity_counts.plot(kind='bar')

    plt.xlabel('Nível de atividade do usuário')
    plt.ylabel('Número de usuários')
    plt.title('Classificação dos usuários por nível de atividade')

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def analise_itens(dataset):

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

    plt.figure(figsize=(10, 6))
    plt.hist(dataset['rating'], bins=20)

    plt.xlabel('Nota')
    plt.ylabel('Frequência')
    plt.title('Distribuição das avaliações')

    plt.tight_layout()
    plt.show()

    item_popularity = dataset['product_id'].value_counts()

    plt.figure(figsize=(10, 6))
    plt.hist(item_popularity.values, bins=50, log=True)

    plt.xlabel('Número de avaliações por item')
    plt.ylabel('Número de itens (log)')
    plt.title('Distribuição da popularidade dos itens')

    plt.tight_layout()
    plt.show()

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


def cruzar_info(dataset):

    user_rating_profile = au.get_user_rating_profile(dataset)
    product_rating_profile = au.get_product_rating_profile(dataset)

    ranking = (
        product_rating_profile
        .sort_values('count_ratings', ascending=False)
        .head(20)
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        ranking['product'].astype(str),
        ranking['count_ratings']
    )

    plt.xlabel('Número de avaliações')
    plt.ylabel('Produto')
    plt.title('Top 20 produtos mais populares')

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    categorias = dataset['main_category'].value_counts().head(15)

    plt.figure(figsize=(10, 6))
    categorias.plot(kind='bar')

    plt.xlabel('Categoria')
    plt.ylabel('Número de avaliações')
    plt.title('Categorias mais avaliadas pelos usuários')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()