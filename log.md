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


### Day 004: December 18, 2020 (Verified)

**Today's Progress**:

I've added a GPG Key to verify my commits.


**Link to work:**


### Day X: October 01, 2020 (Title)

**Today's Progress**:

**Link to work:**
