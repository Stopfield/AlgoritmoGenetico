# Algoritmo Genético
# Representação Binária codificando um número real (decodificação)

from collections.abc import Callable
import math
from Individuo import Individuo
import random
import copy

type FitnessFloat = float


class Ambiente:
    def __init__(
        self,
        population_size: int,
        generation_threshold: int,
        cross_rate: float,
        mutation_rate: float,
        precision: int,
        max: float,
        min: float,
        fitness: Callable[..., float],
    ) -> None:
        self.population_size = population_size
        self.generation_threshold = generation_threshold
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.fitness = fitness
        self.precision = precision
        self.max = max
        self.min = min

        self.num_bits = math.ceil(math.log2((max - min) * 10**precision))
        self.population: list[tuple[Individuo, FitnessFloat | None]] = []
        self.best_individuals = []
        self.binary_matrix = []
        # self.best_individuals = {
        #     "generation": 1,
        #     "individual": 
        # }
        # self.binary_matrix = {
        #     "generation": 10,
        #     "Matrice": [BitArray]
        # }

    def run(self):
        """Executa o algoritmo genético"""
        print("Iniciando algoritmo genético")
        self.population = self.generate_random_population()
        new_population: list[tuple[Individuo, FitnessFloat | None]] = []
        gen = 0
        while gen < self.generation_threshold:
            self.save_info(generation)
            print(f"Geração {gen}")
            # Avalia população antiga
            print("* Avaliando populaçao...")
            self.evaluate_population(self.population)

            # Resetar a nova
            new_population.clear()

            # Gera nova população
            print("* Gerando nova população...")
            while len(new_population) < self.population_size:
                a, b = self.select_couple(self.population)
                new_a, new_b = self.cross_individuals(a[0], b[0])
                new_population.extend(
                    [
                        (new_a, None),
                        (new_b, None),
                    ]
                )

            # A nova população "envelhece". Vira nossa população atual
            self.population = copy.deepcopy(new_population)
            gen += 1
        print(f"Geração {gen}")
        print("* Avaliando populaçao...")
        self.evaluate_population(self.population)
        print("Fim do Algoritmo")

    def generate_random_population(self) -> list[tuple[Individuo, FitnessFloat | None]]:
        """Gera uma população aleatória para o Ambiente"""
        new_population: list[tuple[Individuo, FitnessFloat | None]] = []
        for _ in range(self.population_size):
            new_population.append(
                (
                    Individuo.random(
                        max=self.max, min=self.min, num_bits=self.num_bits
                    ),
                    None,
                )
            )
        assert len(new_population) == self.population_size
        return new_population

    def select_couple(
        self, population: list[tuple[Individuo, FitnessFloat | None]]
    ) -> tuple[
        tuple[Individuo, FitnessFloat | None], tuple[Individuo, FitnessFloat | None]
    ]:
        """Seleciona um casal de indivíduos da roleta"""
        assert len(population) > 0, "Population cannot be empty"
        assert all(
            [i[1] is not None for i in population]
        ), "Population must be evaluated"

        # Calcular o fitness normalizado e acumulado
        norm_fitness: list[FitnessFloat] = []
        accumulated_norm_fitness: list[FitnessFloat] = []

        _, fitness = zip(*population)
        sum_of_fitness = sum(fitness)

        norm_fitness = [f / sum_of_fitness for _, f in population]

        accumulator = 0.0
        for fitness in norm_fitness:
            accumulator += fitness
            accumulated_norm_fitness.append(accumulator)

        if not math.isclose(accumulated_norm_fitness[-1], 1.0, rel_tol=1e-9):
            raise ValueError(
                "Accumulated normalized fitness does not sum to a number close to 1.0"
            )

        # Gerar um número aleatório normalizado e escolhe dois indivíduos diferentes
        a_index = 0
        b_index = 0
        while a_index == b_index:
            a_index = next(
                i
                for i, f in enumerate(accumulated_norm_fitness)
                if f >= random.random()
            )
            b_index = next(
                i
                for i, f in enumerate(accumulated_norm_fitness)
                if f >= random.random()
            )

        # Retorna os indivíduos
        return (population[a_index], population[b_index])

    def evaluate_population(
        self, population: list[tuple[Individuo, FitnessFloat | None]]
    ):
        """Avalia todos os indivíduos da população com o fitness. Só aplica o fitness para cada indivíduo"""
        for i in range(len(population)):
            individual = population[i][0]
            population[i] = (
                individual,
                self.fitness(individual.decode_x(), individual.decode_y()),
            )
        assert all([i[1] is not None for i in population])

    def cross_individuals(self, a: Individuo, b: Individuo) -> tuple[Individuo, Individuo]:
        """Cruza dois indivíduos e gera outro. Aplica Crossing-Over e Mutation"""
        # Aplica a chance de cruzamento
        if random.random() < self.cross_rate:
            # Seleciona os pontos de corte aleatórios (índices do BitArray)
            p1 = 0
            p2 = 0
            while p1 == p2 or p1 > p2:
                p1 = random.randint(0, self.num_bits * 2)
                p2 = random.randint(0, self.num_bits * 2)

            # Faz o crossing nos genes inteiros
            a_whole_gene = a.get_whole_gene()
            b_whole_gene = b.get_whole_gene()
            a_whole_gene[p1:p2], b_whole_gene[p1:p2] = b_whole_gene[p1:p2], a_whole_gene[p1:p2]

            # Gera novos indivíduos
            new_a = Individuo(
                x=a_whole_gene[0:self.num_bits],
                y=a_whole_gene[self.num_bits:],
                max=self.max,
                min=self.min,
            )

            new_b = Individuo(
                x=b_whole_gene[0:self.num_bits],
                y=b_whole_gene[self.num_bits:],
                max=self.max,
                min=self.min,
            )

            return (new_a, new_b)

        return (a, b)

    def mutate_individual(self, a: Individuo):
        """Muta um indivíduo. Aplica a taxa de mutação em cada bit"""
        a_whole_gene = a.get_whole_gene()
        for i, _ in enumerate(a_whole_gene):
            if random.random() <= self.mutation_rate:
                a_whole_gene[i] = not a_whole_gene[i]
        a.x = a_whole_gene[0:self.num_bits].copy()
        a.y = a_whole_gene[self.num_bits:].copy()

    def get_best_individual(self, population: list[tuple[Individuo, FitnessFloat | None]]) -> tuple[Individuo, FitnessFloat | None]:
        assert all(
            [i[1] is not None for i in population]
        ), "Population must be evaluated"
        _, population_fitness = zip(*population)
        individual = max(enumerate(population_fitness), key=lambda x: x[1])
        return self.population[individual[0]]

    def save_info(self, generation: int) -> None:
        """Salva informações para análise. Melhor indivíduo de cada geração e matriz binária."""
        self.get_best_individual(self.population)
        self.best_individuals.append(
            {
                "generation": generation,
                "individual": individual
            }
        )
        if generation // 10 == 0:
            self.binary_matrix.append(
                {
                    "generation": generation,
                    "matrice": [i.get_whole_gene() for i, _ in self.population]
                }
            )
        ...
