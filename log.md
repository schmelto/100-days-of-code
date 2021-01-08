# 100 Days Of Code - Log

### Day 000: December 14, 2020 (Calculator)

**Today's Progress**:
Build an online calculator app with vanilla JavaScipt, HMTL and CSS.

**Link to work:**
* https://github.com/schmelto/calculator


### Day 001: December 15, 2020 (Ideas - Brainstorming)

**Today's Progress**:
I made a little brainstorm session what projects I can do.
Also I created an Ionic Angular App for my first project the NewsApp and created a first call of the news API.

![ideas](./img/2020-12-15-ideas.jpg)
![newsapp](./img/2020-12-15-NewsApp.jpg)

**Link to work:**

* https://github.com/schmelto/NewsApp


### Day 002:  December 16, 2020 (API-Call)

**Today's Progress**:
I have implemented the right route to newsAPI and added top-headlines and everything.
I tried to add also the country but this will be open for tomorrow. Also the searchstring need to be implemented for "everything".

![update](./img/2020-12-16-update.jpg)

**Link to work:**

* https://github.com/schmelto/NewsApp/commit/bb910cbd5b2f5d0d3cd9868cd0a9a5de8f22ffa5


### Day 003:  December 17, 2020 (Event Listener)

**Today's Progress**:
Today I implemented an event listener service in which I can handle events from the app page and trigger methods in the folder page/component.

``` javascript
import { Injectable } from '@angular/core';
import { Observable, Subject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class EventlistenerService {
  private subject = new Subject<any>();

  sendClickEvent() {
    this.subject.next();
  }

  getClickEvent(): Observable<any>{ 
    return this.subject.asObservable();
  }

}
```
Also i implemented a search bar for browsing all news.

![search](./img/2020-12-17-search.jpg)

**Link to work:**

* https://github.com/schmelto/NewsApp/commit/d7c13807ddec65bd8e5dfac7942bc3e6f73b0df0
* https://github.com/schmelto/NewsApp/commit/0ef3ef6d491eaa106ec20788bd349ce2bef3b757


### Day 004: December 18, 2020 (Verified Infinity Scroll and Reload)

**Today's Progress**:

I've added a GPG Key to verify my commits but this did not work as I want it :D. Futher I addad a infinity scroll with dynamic reload and refresh on pull down.

**Link to work:**

* https://github.com/schmelto/NewsApp/commit/8854de26fa913867de7422867b379f9c0cd1959d

### Day 005: December 19, 2020 (Detail-View)

**Today's Progress**:

Today I've added a Detail view the articles in the news app and made a working back button. Further I solved some bugs in the app.

![Detail](./img/2020-12-19-detail.jpg)

**Link to work:**

* https://github.com/schmelto/NewsApp/issues/10


### Day 006: December 20, 2020 (Categories)

**Today's Progress**:

I added category selection to the news app

![Category](./img/2020-12-20-categories.jpg)

**Link to work:**

* https://github.com/schmelto/NewsApp/commit/75c35636c14c062431d7149e3e654db69d3ce515


### Day 007: December 21, 2020 (Selection)

**Today's Progress**:

Today I added lots of select options to the news app and updated the news.service api call.

```javascript
getData(type, country, category, search, page, language, from, to, sortBy) {
  if (type == 'top-headlines') return this.http.get(`${API_URL}/${type}?pageSize=5&page=${page}&country=${country}&category=${category}&apiKey=${API_KEY}`);
  else return this.http.get(`${API_URL}/${type}?pageSize=5&page=${page}&q=${search}&language=${language}&from=${from}&to=${to}&sortBy=${sortBy}&apiKey=${API_KEY}`);     
}
 ```

![Selection](./img/2020-12-21-selection.jpg)

**Link to work:**

* https://github.com/schmelto/NewsApp/commit/db7d1f71c8ca26ec798f6a3779940092e82dcf8c

### Day 008: December 22, 2020 (Release)

**Today's Progress**:

I resolve the last bugs in the news app and tried to release it to my android phone.

**Link to work:**

* https://github.com/schmelto/NewsApp/commit/5ec22a9fd8846a1d82f3bf595588b777002797d1


### Day 009: December 23, 2020 (Release V1.0.0)

**Today's Progress**:

I released version 1.0.0 of the app as signed apk for android and installed the app on my phone.
Since this project is finished for now I have to look for something new for the #100daysofcode challange.
But this is something for tomorrow :) because tomorrow it's the 24th December (Christmas :christmas_tree:).  

**Link to work:**

* https://github.com/schmelto/NewsApp/releases/tag/1.0.0

### Day 010: December 24, 2020 (Found A BUG!!!!)

**Today's Progress**:

Found a bug in the news app and can't find the solution it does not reset the parameters and data after changing one of them after loading page 2 or more... Opened a question on StackOverflow.

![Bug](./img/2020-12-24-bug.jpg)

**Link to work:**

* https://stackoverflow.com/questions/65440704/how-to-wait-for-methods-called-in-a-subscription

### Day 011: December 25, 2020 (Resolved the Bug finally *_*)

**Today's Progress**:

Finally resolved the bug *_* Now lets move on!
Remindner for me dont use the `ngOnInit()`-Function somewhere else!!!

Now I want to go for the webcrawler-project or some data science stuff. Lets get starting with Python for this!
![Python](./img/2020-12-25-python.jpg)

**Link to work:**

* https://github.com/schmelto/NewsApp/commit/4d285cce866103262d5bfb691bbf390d895de647

### Day 012: December 26, 2020 (Tensorflow)

**Today's Progress**:

I started the FreeCodeCamps Mashine Learning Course not initialy the idea behind 100daysofcode but I need those information for some data science stuff programms I want to build :)

**Link to work:**

* https://github.com/schmelto/machine-learning-with-python

### Day 013: December 27, 2020 (Release 1.0.1)

**Today's Progress**:

Little bit lazy today but released v1.0.1 for the news app

![release](./img/2020-12-27-release.jpg)

**Link to work:**

* https://github.com/schmelto/NewsApp/commit/32e81e6c95b66061d315a73da4e3a73dd1cfa9f2

### Day 014: December 28, 2020 (Release 1.0.2 & Machine Learning)

**Today's Progress**:

Today fixt an other bug in the news app and release it to version 1.0.2 :] somehow it's getting better over time.
Futher I look a little bit at the titanic dataset from tensorflow.

```python
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data

dftrain.head()
```
![titanic](./img/2020-12-28-titanic.jpg)

**Link to work:**

* https://github.com/schmelto/NewsApp/commit/bbaa65fe54c788e8cce072c0557231bf0100866c
* https://colab.research.google.com/github/schmelto/machine-learning-with-python/blob/main/Core_Learning_Algorithms.ipynb#scrollTo=CpllWsKIOGOy

### Day 015: December 29, 2020 (Tensorflow Core Algorithms)

**Today's Progress**:

I've learnd something about model prediction with Tensorflow

```python
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
```

**Link to work:**

* https://colab.research.google.com/github/schmelto/machine-learning-with-python/blob/main/Core_Learning_Algorithms.ipynb#scrollTo=CpllWsKIOGOy

### Day 016: December 30, 2020 (More learning)

**Today's Progress**:

Learned something more

**Link to work:**

* https://github.com/schmelto/machine-learning-with-python/commit/f4ab2fbc9e5adacfacdee2cc01c50974e62ad471

### Day 017: December 31, 2020 (Deeplearning with Python - MNIST)

**Today's Progress**:

I created a little model for prediction the MNIST-Dataset.

![mnist](./img/2020-12-31-mnist.jpg)

**Link to work:**

* https://github.com/schmelto/machine-learning-with-python/commit/76ea931acc90d55566214aa961e9f8792e5f695d

### Day 018: Januar 01, 2021 (Fashion MNIST)

**Today's Progress**:

I tried on myself to predict the fashion MNIST dataset.

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

total_classes = 10
train_vec_labels = keras.utils.to_categorical(train_labels, total_classes)
test_vec_labels = keras.utils.to_categorical(test_labels, total_classes)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='sigmoid'), 
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='sgd',
    loss='mean_squared_error',
    metrics=['accuracy'])
    
model.fit(train_images, train_vec_labels, epochs=50, verbose=True)

eval_loss, eval_accuracy = model.evaluate(test_images, test_vec_labels, verbose=False)
print("Model accuracy: %.2f" % eval_accuracy)
```

Further I added a [README](https://github.com/schmelto/NewsApp/commit/d59b4b0f05f71e9bd9756145ca8808c990afd36f) to the NewsApp.

**Link to work:**

* https://github.com/schmelto/machine-learning-with-python/commit/0e1a3763c94e539abccf9641d7f495d832c080f2
* https://github.com/schmelto/NewsApp/blob/master/README.md | https://github.com/schmelto/NewsApp/commit/d59b4b0f05f71e9bd9756145ca8808c990afd36f

### Day 019: Januar 02, 2021 (README)

**Today's Progress**:

I've updated the [README](https://github.com/schmelto/NewsApp#readme) of the [NewsApp](https://github.com/schmelto/NewsApp) and made some cool badges:

![GitHub release (latest by date)](https://img.shields.io/github/v/release/schmelto/NewsApp?style=for-the-badge)

Further I made some small changes in the [machine learning](https://github.com/schmelto/machine-learning) repo. Also I added an introduction to Numpy and evaluated activation functions of the newtworks.


```python
model_relu = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model_linear = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='linear'),
    keras.layers.Dense(10, activation='linear')
])

model_sigmoid = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(10, activation='sigmoid')
])

model_tanh = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='tanh'),
    keras.layers.Dense(10, activation='tanh')
])

models = [model_relu, model_linear,
          model_sigmoid,model_tanh]
```

**Link to work:**

* https://github.com/schmelto/NewsApp/commit/087b74d13fa5288b79992ccfb6cea29097bcc00b
* https://github.com/schmelto/machine-learning/commit/15abacab04630dd2ae377e55046869cd72205b61


### Day 020: Januar 03, 2021 (Numpy)

**Today's Progress**:

Finished the Numpy introduction ([here](https://github.com/schmelto/machine-learning/blob/main/Deeplearning/introduction_to_numpy.ipynb)) and have the idea for the #100daysofcode challenge to build an app that recognize handwriting via camera. Maybe give it a try :D ([here](https://github.com/schmelto/text-recognition))

![text-recognition](./img/2021-01-03-text-recognition.jpg)

**Link to work:**

* https://github.com/schmelto/machine-learning/commit/c689089b36552d9fc560a13d38b8dbd7aedc2bbc

### Day 021: Januar 04, 2021 (Maploitlib)

**Today's Progress**:

I've created an introduction to Maploitlib and plotted some plots :D

![Maploitlib](./img/2021-01-04-Maploitlib.jpg)

**Link to work:**

* https://github.com/schmelto/machine-learning/commit/c8cbc0c68f01c2c2721894fff7bd5980c122529c

### Day 022: Januar 05, 2021 (Keras / Tensorflow / MNIST)

**Today's Progress**:

I take a deeper look at Keras, Tensorflow and analyzed the MNIST dataset.

![MNSIT](./img/2021-01-05-mnist.jpg)

**Link to work:**

* https://github.com/schmelto/machine-learning/commit/dbc504d25c7c2861d0d73dd0cd4f03ab55345dd2

### Day 023: Januar 06, 2021 (Complex layer structure)

**Today's Progress**:

I take a look at complex layer structure.

![layer](./img/2021-01-06-layer.jpg)

**Link to work:**

* https://github.com/schmelto/machine-learning/commit/9f543cb5630bb46a7fe7f908265e636196d79253

### Day 024: Januar 07, 2021 (Loss Functions)

**Today's Progress**:

I optimized the models with loss functions

```python
optimizer = 'sgd'

model_mse.compile(
    optimizer=optimizer,
    loss='mean_squared_error',
    metrics=['accuracy']
)
model_cce.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model_scce.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
**Link to work:**

* https://github.com/schmelto/machine-learning/commit/e8db352a8ce8817df070a6927f737441c4d3b21c

### Day 025: Januar 08, 2021 (Optimizer and Hyperparameters)

**Today's Progress**:

Today I've taked a look on how I can optimize my models wit Optimizer and Hyperparameters.

```python
model.compile(
    optimizer = tf.keras.optimizers.Adam(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```
**Link to work:**

* https://github.com/schmelto/machine-learning/commit/6af1353a3166fb83552b0e3a35a5a93159c9fa16
* https://github.com/schmelto/machine-learning/commit/210d38408c687e533ef50fc5603e7bd2d811acc9
