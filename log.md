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


### Day 006: December 19, 2020 (Categories)

**Today's Progress**:

I added category selection to the news app

![Category](./img/2020-12-20-categories.jpg)

**Link to work:**

* https://github.com/schmelto/NewsApp/commit/75c35636c14c062431d7149e3e654db69d3ce515

