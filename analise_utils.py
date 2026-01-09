import pandas as pnd

def normaliza_0_10(series):
    min_v = series.min()
    max_v = series.max()

    if max_v != min_v:
        return ((series - min_v) / (max_v - min_v)) * 10
    return pnd.Series(5, index=series.index)



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

    users_stats['generoso_ou_critico'] = normaliza_0_10(users_stats['avg_rating'])

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

    product_stats['qualidade'] = normaliza_0_10(product_stats['avg_rating'])

    return product_stats


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