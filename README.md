# Limited Queue

Circular queue that overrides the oldest data if trying to push data when queue is full.

All operations are of **`O(1)`** complexity, except the constructor with `O(Vec::with_capacity)`.

The generic type `T` needs to have `Default` trait for `pop` operation, since we need to replace the popped element with some element.

There is a similar library [`circular-queue`](https://github.com/YaLTeR/circular-queue) I found, but without the basic `peek` and `pop` operations. The comparison for now is listed below:

||[`LimitedQueue`](.)| [`circular-queue`]((https://github.com/YaLTeR/circular-queue)) |
|:-:|:-|:-|
|Algorithm|circular queue (front-rear)|circular queue (based on `len` and `capacity` provided by `Vec`)|
|Element trait bound|`Default`|-|
|`push`, size-related methods|✅|✅|
|`peek`, `pop` support|✅|❌|
|Indexing|✅<br>- front: `0`<br>- rear: `capacity`<br>- optionally mutable|❌|
|Iterator|✅<br>- front to rear|✅<br>- both ways<br>- optionally mutable|
|`clear` complexity|`O(1)`|`O(Vec::clear)`|
|`serde` support|❌ (TODO)|✅|

We welcome any kinds of contributions, please don't be hesitate to submit issues & PRs.