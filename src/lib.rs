//! # LimitedQueue
//!
//! `LimitedQueue<T>` is a limited queue that
//! overrides the oldest data if trying to
//! push a data when the queue is full.
//!
//! All operations are of `O(1)` complexity,
//! except the constructor with `O(Vec::with_capacity)`.

use std::mem::take;

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
}

impl<T: Default> LimitedQueue<T> {
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
            q: Vec::with_capacity(cap + 1),
            front: 0usize,
            rear: 0usize,
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

    /// Pop the first element from queue
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
            Some(ret)
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
            popped = self.pop();
        }
        if self.q.len() != self.q.capacity() {
            self.q.push(ele);
        } else {
            self.q[self.rear] = ele;
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
        self.front == self.rear
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.next_idx(self.rear) == self.front
    }

    #[inline]
    pub fn len(&self) -> usize {
        (self.rear + self.q.capacity() - self.front) % self.q.capacity()
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
    }
}

impl<T> std::ops::Index<usize> for LimitedQueue<T> {
    type Output = T;

    fn index(&self, idx: usize) -> &Self::Output {
        let real_idx = (idx + self.front) % self.q.capacity();
        &self.q[real_idx]
    }
}

impl<T> std::ops::IndexMut<usize> for LimitedQueue<T> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        let real_idx = (idx + self.front) % self.q.capacity();
        &mut self.q[real_idx]
    }
}

pub struct LimitedQueueIterator<'a, T: Default> {
    lq: &'a LimitedQueue<T>,
    idx: usize,
}

impl<'a, T: Default> Iterator for LimitedQueueIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let cur_idx = self.idx;
        if cur_idx == self.lq.len() {
            None
        } else {
            self.idx = self.lq.next_idx(self.idx);
            Some(&self.lq[cur_idx])
        }
    }
}

#[cfg(test)]
mod tests {
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
        for (n, element) in q.iter().enumerate() {
            assert_eq!(element.clone(), q[n]); // 5, 6, 7, 8, 9
            assert_eq!(element.clone(), n + 5); // 5, 6, 7, 8, 9
        }
    }

    #[test]
    fn test_change_size() {
        const MAX_SZ: usize = 100;
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
                        _ => sz == q.len(),
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
    #[should_panic]
    fn test_invalid_indexing() {
        let mut q = LimitedQueue::with_capacity(5);
        q.push(1);
        q[100];
    }
}
