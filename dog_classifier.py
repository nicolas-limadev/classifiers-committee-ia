import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carregar o conjunto de dados
caminho_arquivo = './base/Dog Breads Around The World.csv'
df = pd.read_csv(caminho_arquivo)

# Limpeza de dados: Converter 'Average Weight (kg)' para numérico
def convert_weight(value):
    if isinstance(value, str):
        # Substituir qualquer caractere não dígito, exceto '.' e '-'
        clean_value = ''.join(c if c.isdigit() or c in ['.', '-'] else ' ' for c in value)
        nums = clean_value.strip().split()
        nums = [float(num) for num in nums if num.replace('.', '', 1).replace('-', '', 1).isdigit()]
        if nums:
            return sum(nums) / len(nums)
        else:
            return np.nan
    else:
        return value

df['Average Weight (kg)'] = df['Average Weight (kg)'].apply(convert_weight)
df['Average Weight (kg)'] = pd.to_numeric(df['Average Weight (kg)'], errors='coerce')

# Codificar variáveis categóricas
colunas_categoricas = ['Type', 'Size', 'Grooming Needs', 'Exercise Requirements (hrs/day)',
                       'Intelligence Rating (1-10)', 'Shedding Level', 'Health Issues Risk',
                       'Training Difficulty (1-10)']

label_encoder = LabelEncoder()
for col in colunas_categoricas:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Codificar variável alvo
df['Good with Children'] = df['Good with Children'].apply(lambda x: 1 if x == 'Yes' else 0)

# Tratar valores ausentes
df.dropna(inplace=True)

# Seleção de características
df_features = df.drop(columns=['Name', 'Origin', 'Unique Feature', 'Good with Children'])
target = df['Good with Children']

# Dividir o conjunto de dados
X_train, X_test, y_train, y_test = train_test_split(df_features, target, test_size=0.3, random_state=42)

# Inicializar classificadores
modelos = {
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Árvore de Decisão': DecisionTreeClassifier(max_depth=5, random_state=42)
}

# Treinar classificadores individuais
for nome_modelo, modelo in modelos.items():
    modelo.fit(X_train, y_train)

# Avaliar classificadores individuais
for nome_modelo, modelo in modelos.items():
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred)
    print(f"Resultados para {nome_modelo}:")
    print(f"Acurácia: {acuracia}")
    print(f"Relatório de Classificação:\n{relatorio}\n")

# Criar e avaliar o ensemble
voting_clf = VotingClassifier(
    estimators=[
        ('naive_bayes', modelos['Naive Bayes']),
        ('knn', modelos['K-Nearest Neighbors']),
        ('decision_tree', modelos['Árvore de Decisão'])
    ],
    voting='hard'
)
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
acuracia_ensemble = accuracy_score(y_test, y_pred)
relatorio_ensemble = classification_report(y_test, y_pred)
print("Resultados para o Comitê de Classificadores (Ensemble):")
print(f"Acurácia: {acuracia_ensemble}")
print(f"Relatório de Classificação:\n{relatorio_ensemble}")
