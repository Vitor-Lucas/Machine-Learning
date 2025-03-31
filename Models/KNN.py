import random
from typing import Any, NamedTuple, List
from collections import Counter
from Math.LinearAlg import Vector, distance

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

class KNN:
    def __init__(self, train_to_dataset_ratio:float):
        self.train_labeled_points: List[LabeledPoint] = []
        self.val_labeled_points: List[LabeledPoint] = []
        self.train_to_val_ratio = train_to_dataset_ratio

    def read_dataset(self, path: str):
        dataset = []
        with open(path, 'r') as file:
            for line in file.readlines():
                line = line.replace('\n', '').split(',')
                point = [float(value) for value in line[:-1]]
                label = line[-1]
                dataset.append(LabeledPoint(point, label))

        rand = random.Random()
        for labeled_point in dataset:
            if rand.random() <= self.train_to_val_ratio:
                self.train_labeled_points.append(labeled_point)
            else:
                self.val_labeled_points.append(labeled_point)


    def majority(self, points:list) -> Any:
        counter = Counter(points)
        winner, winner_count = counter.most_common(1)[0]

        num_winners = len([count
                           for count in counter.values()
                           if count == winner_count])
        if num_winners == 1:
            return winner
        return self.majority(points[:-1])

    def classify(self, k:int, new_point: Vector):
        ordered_points = [lb_point
                          for lb_point in sorted(self.train_labeled_points, key= lambda lp: distance(lp.point, new_point))]
        k_nearest_labels = [lb_point.label for lb_point in ordered_points[:k]]

        return self.majority(k_nearest_labels)


if __name__ == '__main__':
    model = KNN(0.80)
    model.read_dataset('/home/vitor/Coding/Python/Machine-Learning/Datasets/iris/Iris_proc.csv')


    train_size = len(model.train_labeled_points)
    val_size = len(model.val_labeled_points)
    print('Dataset info:')
    print(f'Train size: {train_size}')
    print(f'Val size: {val_size}')
    print(f'Train to dataset ratio: {train_size / (train_size + val_size)}')

    right = 0
    wrong = 0
    for entity in model.val_labeled_points:
        pred_label = model.classify(k=5, new_point=entity.point)
        print(f'Predicted label: {pred_label}')
        print(f'Actual label: {entity.label}')

        if pred_label == entity.label:
            right += 1
        else:
            wrong += 1

    print('Model got:')
    print(f'{right} rigths')
    print(f'{wrong} wrongs')