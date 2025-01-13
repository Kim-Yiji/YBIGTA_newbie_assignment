from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        """
        Matrix 클래스의 생성자.

        :param matrix: 2D 리스트 형태의 행렬 데이터를 받아 저장.
        """
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        """
        주어진 값 n으로 채워진 행렬을 생성하여 반환.

        :param n: 행렬을 채울 값.
        :param shape: 행렬의 크기 (행, 열).
        :return: 모든 값이 n인 행렬.
        """
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        """
        모든 요소가 0인 행렬을 생성하여 반환.

        :param shape: 행렬의 크기 (행, 열).
        :return: 모든 값이 0인 행렬.
        """
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        """
        모든 요소가 1인 행렬을 생성하여 반환.

        :param shape: 행렬의 크기 (행, 열).
        :return: 모든 값이 1인 행렬.
        """
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        """
        주어진 크기의 단위 행렬을 생성하여 반환.

        :param n: 행렬의 크기 (행과 열이 같은 크기).
        :return: 단위 행렬 (주대각선이 1이고 나머지 요소는 0).
        """
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        """
        행렬의 크기를 반환하는 프로퍼티.

        :return: (행, 열) 형태의 튜플.
        """
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        """
        행렬의 깊은 복사본을 반환.

        :return: 현재 행렬의 복사본.
        """
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        """
        행렬의 특정 위치에 있는 값을 반환.

        :param key: (행, 열) 튜플.
        :return: 해당 위치의 값.
        """
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        """
        행렬의 특정 위치에 값을 설정.

        :param key: (행, 열) 튜플.
        :param value: 설정할 값.
        """
        i, j = key
        self.matrix[i][j] = value

    def __matmul__(self, matrix: Matrix) -> Matrix:
        """
        두 행렬을 곱셈(행렬 곱)을 수행.

        :param matrix: 곱할 행렬.
        :return: 두 행렬의 곱을 나타내는 새로운 행렬.
        :raise AssertionError: 두 행렬의 열과 행의 크기가 맞지 않으면 예외 발생.
        """
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1  # 열 크기와 행 크기가 일치해야 함

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]
                    result[i, j] %= Matrix.MOD  # MOD 적용

        return result

    def __pow__(self, n: int) -> Matrix:
        """
        행렬의 거듭제곱을 계산 (분할 정복을 사용하여 효율적으로 계산).

        :param n: 거듭제곱할 값.
        :return: 행렬의 n번 거듭제곱.
        """
        result = Matrix.eye(self.shape[0])  # 단위 행렬로 초기화
        base = self.clone()

        while n > 0:
            if n % 2 == 1:
                result = result @ base  # 결과에 base 행렬 곱하기
            base = base @ base  # base 행렬을 제곱
            n //= 2  # n을 반으로 줄임

        return result

    def __repr__(self) -> str:
        """
        행렬을 문자열 형태로 출력.

        :return: 행렬을 보기 좋게 문자열로 변환한 값.
        """
        return "\n".join(" ".join(str(self[i, j]) for j in range(self.shape[1])) for i in range(self.shape[0]))



from typing import Callable
import sys


"""
아무것도 수정하지 마세요!
"""


def main() -> None:
    intify: Callable[[str], list[int]] = lambda l: [*map(int, l.split())]

    lines: list[str] = sys.stdin.readlines()

    N, B = intify(lines[0])
    matrix: list[list[int]] = [*map(intify, lines[1:])]

    Matrix.MOD = 1000
    modmat = Matrix(matrix)

    print(modmat ** B)


if __name__ == "__main__":
    main()