
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib import rc
rc('font', family='AppleGothic')

class Perceptron(object):
    """
    퍼셉트론 클래스
        특징 값 * 해당 가중치의 결과
        입력을 제외하면, 단일 계층만이 존재함
        활성화 함수도 없음
    속성
        lr          : 학습률
        n_iter      : 학습 횟수
        random_state: 난수 시드

    메서드
        fit         : 학습 수행
        net_input   : 뉴럴넷 입력
        predict     : 뉴럴넷 입력 + 결과
    """

    def __init__(self, lr=0.01, n_iter=50, random_state=1):
        self.lr             = lr
        self.n_iter         = n_iter
        self.random_state   = random_state

    def fit(self, X, y):
        """
            X는 입력 값: 모양새는 (샘플 수, 특징 수)
        """

        # 난수 발생 시드를 설정,
        # 정규분포(rgen.normal)에서 항상 동일한 난수를 가져오기 위함
        rgen            = np.random.RandomState(self.random_state)

        # 가중치 행렬,
        # loc: 중앙 값, scale: 표준 편차(std), size: 생성할 난수의 개수
        #
        #   X.shape[1]  : 특징 수, 
        #   +1          : 편향 (bias)
        #   (특징 수 + 1)와 동일한 개수의 가중치가 존재함
        self.w_         = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_    = []

        # 학습 루프
        # n_iter 만큼 반복
        for _ in range(self.n_iter):
            errors = 0

            # zip 으로 X, y를 묶어서, 튜플 형태로 루프를 돈다
            # 즉, 모든 (입력, 레이블) 페어가 대상이 됨
            for xi, target in zip(X, y):

                # 가중치 업데이트의 정도를 계산
                update      = self.lr * (target - self.predict(xi))

                # w_[0] 은 bias에 대한 것으로 x값이 실질적으로 0임
                #   따라서, update 만을 더해주면 됨
                # w_[1:]은 bias를 제외한 모든 가중치에 대한 것임 (w_[1], w_[2], ..., w_[n])
                #   따라서, xi에 update를 곱해줘야함
                self.w_[1:] = self.w_[1:] + (xi * update)
                self.w_[0]  = self.w_[0] + update

                # 여기서 에러란, 가중치 업데이트가 발생하지 않는 경우를 카운트한 것
                # 가중치 업데이트가 없다는 것은 학습이 필요 없다는 의미
                # 즉, errors는 가중치 업데이트가 필요했던 (학습이 필요했던) 것을 카운트
                errors      = errors + int(update != 0.0)

            self.errors_.append(errors)
        
        return self

    def net_input(self, X):
        # 단순, 입력과 가중치의 곱셈을 수행
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # np.where는 특정 조건에 맞는 경우에 대해, 배열의 값을 치환
        #   일종의 삼항 연산자 같은것
        #   첫 번째 인자가 조건
        #   두 번째 인자는 조건에 맞는 경우 치환값
        #   세 번째 인자는 조건에 맞지 않는 경우 치환값

        return np.where(self.net_input(X) >= 0.0, 1, -1)

def get_iris_dataset_link():
    url = os.path.join('https://archive.ics.uci.edu',
                       'ml', 'machine-learning-databases',
                       'iris', 'iris.data')
    print('URL:', url)
    return url

def get_100_iris_data(iris_df):
    """
    IRIS 데이터는 총 5개의 열을 가진다.
        1~4 열은 특징 값을 위한 것이고, 5 열은 레이블 값을 위한 것이다

    iloc[0:100, 4] => 0~99 (100개) 데이터 중 5 열만을 들고온다
        즉, 레이블만을 들고오는 것임
        values는 numpy 형식의 배열로 치환

        레이블 중 "Iris-setosa" 값을 가지는 행은 -1, 
                "Iris-setosa" 값이 아닌 경우는 1로 치환        
    
    iloc[0:100, [0, 2]] => 0~99 (100개) 데이터 중 1, 3 열만을 들고온다
    """
    y = iris_df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    X = iris_df.iloc[0:100, [0, 2]].values

    return X, y

def plot_iris_data(X, y):
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='x', label='versicolor')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal legnth [cm]')
    plt.legend(loc='upper left')
    plt.show()

def train(X, y):
    ppn = Perceptron(lr=0.1, n_iter=10)
    ppn.fit(X, y)
    return ppn

def plot_errors(errors):
    plt.plot(range(1, len(errors) + 1),
             errors, marker='o')
    plt.xlabel('학습 횟수')
    plt.ylabel('업데이트가 발생한(학습이 필요했던) 횟수')
    plt.show()

def plot_decision_regions(X, y, model, resolution=0.02):
    markers = ('s'  , 'x'   , 'o'           , '^'       , 'v')
    colors  = ('red', 'blue', 'lightgreen'  , 'gray'   , 'cyan')
    cmap    = ListedColormap(colors[:len(np.unique(y))])

    x1_min = X[:, 0].min() - 1 
    x1_max = X[:, 0].max() + 1
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0],
                    y=X[y==cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='black')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

def main():
    """
    IRIS 데이터셋(CSV) 의 링크를 가져와서
    이를 직접적으로 pandas의 DataFrame으로 생성

    IRIS 데이터셋은 50개 단위로, 서로 다른 IRIS 종의 꽃 데이터를 가지고 있음
    """
    url     = get_iris_dataset_link()
    iris_df = pd.read_csv(url, header=None, encoding='utf-8')
    print(iris_df.tail())

    """
    처음 100개의 IRIS 데이터를 가져온 다음 그래프를 그린다.
    50개씩 한 IRIS 꽃의 종에 대한 데이터 이기 때문에,
    2 종의 IRIS 꽃이 존재함
    """
    X, y = get_100_iris_data(iris_df)
    plot_iris_data(X, y)

    """
    Perceptron 모델로 학습 수행
    """
    model = train(X, y)
    plot_errors(model.errors_)

    plot_decision_regions(X, y, model)

if __name__ == "__main__":
    main()