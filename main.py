import dados_analise as da
import analise_utils as au

DATASET_PATH = "amazon.csv"

def main():

    dataset = au.preprocess_dataset(DATASET_PATH)

    da.analise_geral(dataset)
    da.analise_usuarios(dataset)
    da.analise_itens(dataset)
    da.analise_interacoes(dataset)

main()
