# Limited Queue

[![Crates.io Version](https://img.shields.io/crates/v/limited-queue.svg)](https://crates.io/crates/limited-queue)
![GitHub top language](https://img.shields.io/github/languages/top/Shiritai/limited-queue)
![GitHub CI Status](https://img.shields.io/github/actions/workflow/status/Shiritai/limited-queue/.github/workflows/ci.yml)
[![Crates.io Downloads](https://img.shields.io/crates/d/limited-queue.svg)](https://crates.io/crates/limited-queue)
[![License](https://img.shields.io/github/license/Shiritai/limited-queue)](LICENSE)

A circular queue that overrides the oldest data if trying to push a data when the queue is full.

All operations are of **`O(1)`** complexity, except the constructor with `O(Vec::with_capacity)`.

The optional method `pop` is provided when `T` satisfies trait bound `Default`.

## Comparison

There is a similar library [`circular-queue`](https://github.com/YaLTeR/circular-queue) I found, but without the basic `peek` and `pop` operations. The comparison for now is listed below:

||[`LimitedQueue`](https://github.com/Shiritai/limited-queue)| [`circular-queue`](https://github.com/YaLTeR/circular-queue) |
|:-:|:-|:-|
|Algorithm|circular queue (front-rear, without additional element slot)|circular queue (based on `len` and `capacity` provided by `Vec`)|
|Element trait bound needed|No, optionally `Default` for `pop` method|-|
|`push`, size-related methods|✅|✅|
|`peek`, `pop` support|✅: `peek`<br>✅: `pop` for `T: Default`|❌|
|Indexing|✅<br>- `[0, len)`<br>- support `[idx]`<br>- support `get(idx)`<br>- optionally mutable (`[idx]`)|❌|
|Iterator|✅<br>- front to rear|✅<br>- both ways<br>- optionally mutable|
|`clear` complexity|`O(1)`|`O(Vec::clear)`|
|`serde` support|❌ (TODO)|✅|

We welcome any kinds of contributions, please don't be hesitate to submit issues & PRs.

## Setup

Please run `scripts/setup.sh` to setup for committing. Currently, the script registers a git pre-commit hook.