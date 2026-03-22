import dados_analise as da
import analise_utils as au

DATASET_PATH = "amazon.csv"


def gerar_dataset_final(dataset):

    user_profile = au.get_user_rating_profile(dataset)
    product_profile = au.get_product_rating_profile(dataset)

    final_dataset = dataset.merge(
        user_profile,
        left_on='user_id',
        right_on='user',
        how='left'
    )

    final_dataset = final_dataset.merge(
        product_profile,
        left_on='product_id',
        right_on='product',
        how='left'
    )

    final_dataset.drop(columns=['user', 'product'], inplace=True)

    return final_dataset


def main():

    dataset = au.preprocess_dataset(DATASET_PATH)

    da.analise_geral(dataset)
    da.analise_usuarios(dataset)
    da.analise_itens(dataset)
    da.analise_interacoes(dataset)
    da.cruzar_info(dataset)

    final_dataset = gerar_dataset_final(dataset)
    final_dataset.to_csv("final_dataset.csv", index=False)

main()
