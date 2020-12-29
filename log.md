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
