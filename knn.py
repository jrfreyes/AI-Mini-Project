
# https://medium.com/analytics-vidhya/image-classification-using-machine-learning-support-vector-machine-svm-dc7a0ec92e01
# Modified from code by Vegi Shanmukh

import pandas as pd
import os
from skimage.transform import resize  # conda install scikit-image
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
from datetime import timedelta

DATASET_PATH = './dataset'
N_CATEGORIES = 14
CATEGORIES = os.listdir(DATASET_PATH)[:N_CATEGORIES]
DIMENSIONS = (50, 50, 3)


def main():
    print(CATEGORIES, len(CATEGORIES), sep='\n')

    flat_data_arr = []
    target_arr = []

    for category in CATEGORIES:
        print(f'Loading... Category: {category}')

        path = os.path.join(DATASET_PATH, category)

        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, DIMENSIONS)
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(CATEGORIES.index(category))

        print(f'Loaded category: {category} successfully')

    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)

    df = pd.DataFrame(flat_data)
    df['Target'] = target

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    param_grid = {
        'n_neighbors': [3, 5, 11, 19],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }

    model = GridSearchCV(KNeighborsClassifier(),
                         param_grid)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.20,
                                                        random_state=77,
                                                        stratify=y)

    print('Split successful')

    print('Training model...')
    start = timer()
    model.fit(x_train, y_train)
    end = timer()
    elapsed = timedelta(seconds=(end - start))
    print(f'Model training successful. Time elapsed {elapsed}')

    print('Testing Data...')
    start = timer()
    y_pred = model.predict(x_test)
    end = timer()
    elapsed = timedelta(seconds=(end - start))
    print(f'Test completed in {elapsed}')
    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(y_test))
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")


if __name__ == '__main__':
    main()



