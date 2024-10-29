from Ambiente import Ambiente
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def F6(x: float, y: float) -> float:
    """Calcula a função Schaffer's F6 dada por:
    F6(x, y) = 0.5 - ((sin(sqrt(x^2 + y^2)))^2 - 0.5) / ((1 + 0.001 * (x^2 + y^2))^2)
    """
    numerator = (math.sin(math.sqrt(x**2 + y**2))) ** 2 - 0.5
    denominator = (1 + 0.001 * (x**2 + y**2)) ** 2
    return 0.5 - (numerator / denominator)

def plot_best_individuals(data):
    df = pd.DataFrame(data)

    # Criando o gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(df['generation'], df['fitness'], marker='o', linestyle='-', color='b')

    # Adicionando títulos e rótulos
    plt.title('Evolução dos Melhores Indivíduos')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.grid()

    # Exibindo o gráfico
    plt.show()

def plot_binary_matrix(data):
    df = pd.DataFrame(data)

    # Definindo o número de subplots
    num_matrices = len(df)
    rows = (num_matrices + 1) // 2  # Número de linhas
    cols = 2  # Número de colunas

    # Criando a figura e os eixos com tamanho ajustado
    fig, axs = plt.subplots(rows, cols, figsize=(8, rows * 3))

    # Iterando sobre as matrizes e plotando
    for index, row in enumerate(df.iterrows()):
        generation = row[1]['generation']
        matrix = np.array([list(map(int, x)) for x in row[1]['matrice']])
        
        # Calculando a posição do subplot
        ax = axs[index // cols, index % cols]
        ax.imshow(matrix, cmap='gray', interpolation='nearest')
        ax.set_title(f'Generation {generation}', fontsize=10)
        ax.set_xlabel('Columns', fontsize=8)
        ax.set_ylabel('Rows', fontsize=8)
        ax.axis('off')  # Desligando os eixos

    # Ajustando o layout
    plt.tight_layout()
    plt.show()

def gemini(data):
    # Carregar os dados
    df = pd.DataFrame(data)

    # Função para converter uma string binária em uma matriz NumPy
    def binary_to_matrix(binary_string):
        return np.array(list(map(int, binary_string))).reshape(1, -1)

    # Criar uma lista de matrizes para cada geração
    matrices = []
    for _, row in df.iterrows():
        generation_matrices = [binary_to_matrix(s) for s in row['matrice']]
        matrices.append(generation_matrices)

    # Função para atualizar o plot a cada frame
    def update(frame):
        plt.imshow(matrices[frame], cmap='gray', interpolation='none')
        plt.title(f'Generation {df["generation"][frame]}')

    # Criar a figura e o plot
    fig, ax = plt.subplots()

    # Criar a animação
    ani = animation.FuncAnimation(fig, update, frames=len(matrices), interval=500)

    # Mostrar a animação
    plt.show()

if __name__ == '__main__':
    algoritmo = Ambiente(
        population_size=200,
        generation_threshold=100,
        cross_rate=0.85,
        mutation_rate=0.01,
        min =-10,
        max = 10,
        precision = 4,
        fitness=F6,
    )

    algoritmo.run()
    i, f = algoritmo.get_best_individual(algoritmo.population)

    print(i)
    print(f"X = {i.decode_x()}")
    print(f"Y = {i.decode_y()}")
    print(f"Fitness: {f}")

    # Análise dos dados
    best_individuals = pd.DataFrame(algoritmo.best_individuals)

    binary_matrix = pd.DataFrame(algoritmo.binary_matrix)

    with open('binary_matrix.csv', '+w') as file:
        binary_matrix.to_csv(file)

    with open('best_individuals.csv', '+w') as file:
        best_individuals.to_csv(file)

    plot_best_individuals(best_individuals)
    plot_binary_matrix(binary_matrix)
