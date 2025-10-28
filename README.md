# Limited Queue

[![Crates.io Version](https://img.shields.io/crates/v/limited-queue.svg)](https://crates.io/crates/limited-queue)
![GitHub top language](https://img.shields.io/github/languages/top/Shiritai/limited-queue)
![GitHub CI Status](https://img.shields.io/github/actions/workflow/status/Shiritai/limited-queue/.github/workflows/ci.yml)
[![Crates.io Downloads](https://img.shields.io/crates/d/limited-queue.svg)](https://crates.io/crates/limited-queue)
[![License](https://img.shields.io/github/license/Shiritai/limited-queue)](LICENSE)

A circular queue that overrides the oldest data if trying to push a data when the queue is full.

All operations are of **`O(1)`** complexity, except the constructor with `O(Vec::with_capacity)`.

## Comparison

There is a similar library [`circular-queue`](https://github.com/YaLTeR/circular-queue) I found, but without the basic `peek` and `pop` operations.
The comparison for now is listed below:

||[`LimitedQueue`](https://github.com/Shiritai/limited-queue)| [`circular-queue`](https://github.com/YaLTeR/circular-queue) |
|:-:|:-|:-|
|Algorithm|circular queue (front-rear, without additional element slot)|circular queue (based on `len` and `capacity` provided by `Vec`)|
|Element trait bound needed|No|-|
|`push`, size-related methods|✅|✅|
|`peek`, `pop` support|✅: `peek`<br>✅: `pop`|❌|
|Indexing|✅<br>- `[0, len)`<br>- support `[idx]`<br>- support `get(idx)`<br>- optionally mutable (`[idx]`)|❌|
|Iterator|✅<br>- `iter()` (DoubleEndedIterator)<br>- `iter_mut()` (DoubleEndedIterator)|✅<br>- both ways<br>- optionally mutable|
|`clear` complexity|`O(1)`|`O(Vec::clear)`|
|`serde` support|❌ (TODO)|✅|

We welcome any kinds of contributions, please don't be hesitate to submit issues & PRs.

## Changelog

### Version 0.2.0 (2025-10-28)

This is a significant feature and stability release.

* **BREAKING (but good) CHANGE:** `pop()` no longer requires the `T: Default` trait bound. This was achieved by refactoring the internal storage to `Vec<Option<T>>` and is verified by tests .
* **New Feature:** Added `iter_mut()`, providing a mutable, double-ended iterator.
* **Enhancement:** `iter()` now implements `DoubleEndedIterator`, allowing for reverse iteration (e.g., `.iter().rev()`).
* **Internal Refactor:** `push()` and `with_capacity()` logic was rewritten to pre-allocate the internal `Vec` with `None`, simplifying the implementation and removing potential panic paths.
* **Fix:** Corrected a logic bug in `get(idx)` that caused incorrect `None` returns.
* **Testing:** Test suite was significantly expanded to cover all methods, edge cases (like capacity 0 and 1), and `panic` conditions, achieving 100% test coverage.

## Setup

Please run `scripts/setup.sh` to setup for committing. Currently, the script registers a git pre-commit hook.