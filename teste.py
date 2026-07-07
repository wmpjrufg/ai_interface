import pandas as pd
import dill as pickle
from sklearn.metrics import r2_score

def principal():
    print("Iniciando a validação do modelo de drogas...")

    # Carregar o dataset
    caminho_csv = 'dados/drug200.csv'
    try:
        df = pd.read_csv(caminho_csv)
        print(f"Dataset carregado. Linhas: {len(df)}")
    except FileNotFoundError:
        print(f"ERRO: Não encontrei o arquivo em {caminho_csv}")
        return

    # Mapeamento
    mapeamento_drug = {
        'DrugY': 0, 'drugC': 1, 'drugX': 2, 'drugA': 3, 'drugB': 4
    }
    df['Drug_num'] = df['Drug'].map(mapeamento_drug)

    # Carregar o modelo
    caminho_modelo = 'dados/reg_gradient_boosting.pkl'
    try:
        with open(caminho_modelo, 'rb') as f:
            modelo = pickle.load(f)
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"ERRO ao carregar o modelo: {e}")
        return

    #  Tratamento dos dados
    # É CRUCIAL que o tratamento do X seja idêntico ao que você faz no app.py
    # Se você usou get_dummies no app.py, tem que usar exatamente igual aqui
    X_input = df.drop(columns=['Drug', 'Drug_num'])
    X_input = pd.get_dummies(X_input, drop_first=True)

    # Previsão
    try:
        previsao = modelo.predict(X_input)
        df['predict'] = previsao
    except Exception as e:
        print(f"ERRO durante a previsão: {e}")
        return

    # Cálculo do R²
    r2 = r2_score(df['Drug_num'], df['predict'])
    print(f"\nO R² do modelo neste dataset é: {r2:.4f}")

    # Exportar
    # df.to_excel('machine_learning_r02.xlsx', index=False)
    # print("Resultados exportados para 'machine_learning_r02.xlsx'")

    # Resumo visual
    print("\nResumo da comparação:")
    print(df)

if __name__ == "__main__":
    principal()