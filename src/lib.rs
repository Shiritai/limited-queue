//! # LimitedQueue
//!
//! `LimitedQueue<T>` is a limited queue that
//! overrides the oldest data if trying to
//! push a data when the queue is full.
//!
//! All operations are of `O(1)` complexity,
//! except the constructor with `O(Vec::with_capacity)`.

use std::mem::{replace, take};

/// A circular queue that overrides the oldest data
/// if trying to push data when queue is full.
///
/// All operations are of `O(1)` complexity,
/// except the constructor with `O(Vec::with_capacity)`.
///
/// # Example
///
/// ```
/// let mut q = limited_queue::LimitedQueue::with_capacity(5);
/// // push_ret: [None x 5, 0, 1, ..., 4]
/// let push_ret = [[None; 5], core::array::from_fn(|i| Some(i))].concat();
/// for (i, pr) in (0..10).zip(push_ret) {
///     assert_eq!(q.push(i), pr);
/// }
/// for (n, element) in q.iter().enumerate() {
///     assert_eq!(element.clone(), q[n]); // 5, 6, 7, 8, 9
///     assert_eq!(element.clone(), n + 5); // 5, 6, 7, 8, 9
/// }
/// ```
///
/// # Error
///
/// For indexing, no bound check will occur, so please check
/// the size of the queue with `len` method before subscription.
///
/// If you need boundary check, please use `get` method.
#[derive(Debug)]
pub struct LimitedQueue<T> {
    q: Vec<T>,
    front: usize,
    rear: usize,
    sz: usize,
}

impl<T> LimitedQueue<T> {
    /// Vec-like constructor
    ///
    /// ```
    /// use limited_queue::LimitedQueue;
    ///
    /// let mut q = LimitedQueue::with_capacity(2);
    ///
    /// assert_eq!(q.push(1), None);
    /// assert_eq!(q.push(2), None);
    ///
    /// // first element popped since the capacity is 2
    /// assert_eq!(q.push(3), Some(1));
    ///
    /// assert_eq!(q.peek(), Some(&2));
    /// assert_eq!(q.pop(), Some(2));
    /// assert_eq!(q.peek(), Some(&3));
    /// assert_eq!(q.pop(), Some(3));
    /// ```
    ///
    /// @param `cap` Capacity (limit size) of the queue
    #[inline]
    pub fn with_capacity(cap: usize) -> LimitedQueue<T> {
        LimitedQueue {
            q: Vec::with_capacity(cap),
            front: 0usize,
            rear: 0usize,
            sz: 0usize,
        }
    }

    /// Get the element at position `idx`,
    /// a.k.a. the position from the start of queue
    ///
    /// ```
    /// use limited_queue::LimitedQueue;
    ///
    /// let mut q = LimitedQueue::with_capacity(2);
    /// q.push(1);
    /// assert_eq!(q.get(100), None);
    /// ```
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&T> {
        if idx >= self.front {
            None
        } else {
            Some(&self[idx])
        }
    }

    /// Peek the oldest element at the front of the queue
    ///
    /// ```
    /// use limited_queue::LimitedQueue;
    ///
    /// let mut q = LimitedQueue::with_capacity(1);
    ///
    /// q.push(1234);
    /// assert_eq!(q.peek(), Some(&1234));
    /// assert_eq!(q.pop(), Some(1234));
    /// assert_eq!(q.peek(), None);
    /// ```
    #[inline]
    pub fn peek(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            Some(&self.q[self.front])
        }
    }

    /// Push a new element into queue,
    /// removing the oldest element if the queue is full
    #[inline]
    pub fn push(&mut self, ele: T) -> Option<T> {
        let mut popped = None;
        if self.is_full() {
            // popped = self.pop();
            // use `replace` so no implicit `Default` will be called
            popped = Some(replace(&mut self.q[self.rear], ele));
            // and move forth the front idx to simulate `pop` operation
            self.front = self.next_idx(self.front);
        } else {
            if self.q.len() == self.rear && self.q.len() < self.q.capacity() {
                // if the vector is shorter than capacity
                self.q.push(ele);
            } else if self.q.len() > self.rear {
                self.q[self.rear] = ele;
            } else {
                panic!("[limited-queue::push] Error, bad push position");
            }
            self.sz += 1;
        }
        self.rear = self.next_idx(self.rear);
        popped
    }

    /// Inner method: next index of the inner vector
    #[inline]
    fn next_idx(&self, idx: usize) -> usize {
        (idx + 1) % self.q.capacity()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sz == 0
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.sz == self.q.capacity()
    }

    /// Get queue length
    ///
    /// ```
    /// use limited_queue::LimitedQueue;
    ///
    /// let mut q = LimitedQueue::with_capacity(3);
    ///
    /// q.push(1234);
    /// assert_eq!(q.len(), 1);
    /// q.push(1234);
    /// assert_eq!(q.len(), 2);
    /// q.push(1234);
    /// assert_eq!(q.len(), 3);
    /// q.push(1234);
    /// assert_eq!(q.len(), 3);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.sz
    }

    /// To traverse all the elements in
    /// `LimitedQueue`, for example:
    ///
    /// ```
    /// use limited_queue::LimitedQueue;
    ///
    /// let mut q = LimitedQueue::with_capacity(5);
    /// for i in 0..10 {
    ///     q.push(i);
    /// }
    /// for (&n, element) in q.iter().zip(5usize..=9) {
    ///     // will be 5, 6, 7, 8, 9 since 0 ~ 4
    ///     // are popped because the queue is full
    ///     assert_eq!(element.clone(), n);
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> LimitedQueueIterator<T> {
        LimitedQueueIterator { lq: self, idx: 0 }
    }

    // #[inline]
    // pub fn iter_mut(&self) -> LimitedQueueIterator<T> {
    //     LimitedQueueIterator {
    //         lq: self,
    //         idx: self.front,
    //     }
    // }

    /// `O(1)` method to (lazily) clear all the elements
    ///
    /// ```
    /// use limited_queue::LimitedQueue;
    ///
    /// let mut q = LimitedQueue::with_capacity(5);
    /// for i in 0..10 {
    ///     q.push(i);
    /// }
    /// q.clear();
    /// assert_eq!(q.peek(), None);
    /// assert_eq!(q.is_empty(), true);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.front = 0;
        self.rear = 0;
        self.sz = 0;
    }

    /// private method of boundary check and indexing
    #[inline]
    fn indexing(&self, idx: usize) -> usize {
        if idx >= self.sz {
            panic!("Invalid subscription index: {}", idx)
        }
        (idx + self.front) % self.q.capacity()
    }
}

/// Optionally provide `pop` method for
/// types that implements `Default` trait
impl<T: Default> LimitedQueue<T> {
    /// Pop the first element from queue,
    /// will replace the element in queue with
    /// the `Default` value of the element
    ///
    /// ```
    /// use limited_queue::LimitedQueue;
    ///
    /// let mut q = LimitedQueue::with_capacity(1);
    ///
    /// q.push(1234);
    /// assert_eq!(q.pop(), Some(1234));
    /// assert_eq!(q.pop(), None);
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let ret = take(&mut self.q[self.front]);
            self.front = self.next_idx(self.front);
            self.sz -= 1;
            Some(ret)
        }
    }
}

impl<T> std::ops::Index<usize> for LimitedQueue<T> {
    type Output = T;

    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        let real_idx = self.indexing(idx);
        &self.q[real_idx]
    }
}

impl<T> std::ops::IndexMut<usize> for LimitedQueue<T> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        let real_idx = self.indexing(idx);
        &mut self.q[real_idx]
    }
}

pub struct LimitedQueueIterator<'a, T> {
    lq: &'a LimitedQueue<T>,
    idx: usize,
}

impl<'a, T> Iterator for LimitedQueueIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let cur_idx = self.idx;
        if cur_idx == self.lq.len() {
            None
        } else {
            self.idx += 1;
            Some(&self.lq[cur_idx])
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic;

    use rand::Rng;

    use crate::LimitedQueue;

    #[test]
    fn test_iter() {
        let mut q = crate::LimitedQueue::with_capacity(5);
        // push_ret: [None x 5, 0, 1, ..., 4]
        let push_ret = [[None; 5], core::array::from_fn(|i| Some(i))].concat();
        for (i, pr) in (0..10).zip(push_ret) {
            assert_eq!(q.push(i), pr);
        }
        assert_eq!(q.len(), 5);
        for (n, element) in q.iter().enumerate() {
            assert_eq!(element.clone(), q[n]); // 5, 6, 7, 8, 9
            assert_eq!(element.clone(), n + 5); // 5, 6, 7, 8, 9
        }
    }

    #[test]
    fn test_change_size() {
        const MAX_SZ: usize = 25;
        let mut q: LimitedQueue<i32> = crate::LimitedQueue::with_capacity(MAX_SZ);
        let mut sz = 0;
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let op = rng.gen_range(0..=2);
            match op {
                0 => {
                    if q.push(rng.gen()).is_none() {
                        sz += 1
                    };
                }
                1 => {
                    if q.pop().is_some() {
                        sz -= 1
                    };
                }
                _ => {
                    assert!(match sz {
                        0 => q.is_empty() && q.pop().is_none() && q.peek().is_none(),
                        MAX_SZ => q.is_full(),
                        _ => sz == q.len() && sz < MAX_SZ,
                    });
                }
            };
        }
    }

    #[test]
    #[should_panic]
    fn test_zero_len_invalid_indexing() {
        LimitedQueue::with_capacity(0)[0]
    }

    #[test]
    fn test_invalid_indexing() {
        // shadow out panic message for unwind in the loop
        let old_hook = panic::take_hook();
        panic::set_hook(Box::new(|_info| {}));

        let mut q = LimitedQueue::with_capacity(5);
        q.push(1);
        for i in 5..100 {
            let invalid_access = || q[i];
            let should_be_false = panic::catch_unwind(invalid_access).is_err();
            if !should_be_false {
                // reset panic hook to show error message
                panic::set_hook(old_hook);
                // panic with reason
                panic!("Indexing with idx: {} cannot trigger panic.", i);
            }
        }

        panic::set_hook(old_hook);
    }

    #[test]
    fn test_clear() {
        let mut q = LimitedQueue::with_capacity(10);
        for _ in 0..3 {
            q.push(1);
        }
        assert_eq!(q.len(), 3);
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.len(), 2);
        q.clear();
        assert_eq!(q.len(), 0);
        for _ in 0..3 {
            q.push(1);
        }
        assert_eq!(q.len(), 3);
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.len(), 2);
        q.clear();
        assert_eq!(q.len(), 0);
    }
}
