# Binária codificando um número real (decodificação)

from bitstring import BitArray
import random
import math

class Individuo:
    num_individuos = 0
    def __init__(self, x: BitArray, y: BitArray, min: float, max: float) -> None:
        self.x = x
        self.y = y
        self.min = min
        self.max = max

        self.label = Individuo.num_individuos
        Individuo.num_individuos += 1
    
    def decode_x(self) -> int:
        return self.x.uint * (self.max - self.min) / (2 ** self.x.length - 1) + self.min
    
    def decode_y(self) -> int:
        return self.y.uint * (self.max - self.min) / (2 ** self.y.length - 1) + self.min
    
    def get_whole_gene(self) -> BitArray:
        whole_gene = self.x.copy()
        whole_gene.append(self.y.copy())
        num_bits = self.x.length
        assert len(whole_gene) == num_bits * 2 \
            and whole_gene[0:num_bits] == self.x \
            and whole_gene[num_bits:] == self.y, \
            "Whole gene is not equal to individual genes"
        return whole_gene
    
    @staticmethod
    def random(min: float, max: float, num_bits: int) -> 'Individuo': # type: ignore
        return Individuo(
            x = BitArray(uint=random.randint(0, 2**num_bits - 1), length=num_bits),
            y = BitArray(uint=random.randint(0, 2**num_bits - 1), length=num_bits),
            min = min,
            max = max,
        )
    
    @staticmethod
    def copy(i: "Individuo") -> "Individuo":
        return Individuo(
            x = i.x.copy(),
            y = i.y.copy(),
            max = i.max,
            min = i.min
        )


    def __repr__(self) -> str:
        return f'Indivíduo {self.label}'

# Definição do Indivíduo
max = 10
min = -10
precision = 4
num_bits = math.ceil(math.log2((max - min) * 10 ** precision))

a = Individuo(
    x=BitArray(uint=0, length=num_bits),
    y=BitArray(uint=0, length=num_bits),
    min=min,
    max=max,
)