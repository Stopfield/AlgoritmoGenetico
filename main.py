from Ambiente import Ambiente
import math

def F6(x: float, y: float) -> float:
    """Calcula a função Schaffer's F6 dada por:
    F6(x, y) = 0.5 - ((sin(sqrt(x^2 + y^2)))^2 - 0.5) / ((1 + 0.001 * (x^2 + y^2))^2)
    """
    numerator = (math.sin(math.sqrt(x**2 + y**2))) ** 2 - 0.5
    denominator = (1 + 0.001 * (x**2 + y**2)) ** 2
    return 0.5 - (numerator / denominator)

if __name__ == '__main__':
    algoritmo = Ambiente(
        population_size=5,
        generation_threshold=100,
        cross_rate=0.85,
        mutation_rate=0.01,
        min =-10,
        max = 10,
        precision = 4,
        fitness=F6,
    )

    # p = algoritmo.generate_random_population()
    # algoritmo.evaluate_population(p)
    # a, b = algoritmo.select_couple(p)
    # new_a, new_b = algoritmo.cross_individuals(a[0], b[0])

    # print(new_b.get_whole_gene().pp())
    # algoritmo.mutate_individual(new_b)
    # print(new_b.get_whole_gene().pp())
    algoritmo.run()
    i, f = algoritmo.get_best_individual(algoritmo.population)

    print(i)
    print(f"X = {i.decode_x()}")
    print(f"Y = {i.decode_y()}")
    print(f"Fitness: {f}")

