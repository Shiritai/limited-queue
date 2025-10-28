//! # LimitedQueue
//!
//! `LimitedQueue<T>` is a limited queue that
//! overrides the oldest data if trying to
//! push a data when the queue is full.
//!
//! All operations are of `O(1)` complexity,
//! except the constructor with `O(Vec::with_capacity)`.

use std::{
    marker::PhantomData,
    mem::{replace, take},
};

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
    q: Vec<Option<T>>,
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
        if cap == 0 {
            panic!("Cannot create a LimitedQueue with zero capacity");
        }
        let mut q = Vec::with_capacity(cap);
        q.resize_with(cap, Option::default);

        LimitedQueue {
            q,
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
        if idx >= self.sz {
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
            self.q[self.front].as_ref()
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
            popped = replace(&mut self.q[self.rear], Some(ele));
            // and move forth the front idx to simulate `pop` operation
            self.front = self.next_idx(self.front);
        } else {
            let _ = std::mem::replace(&mut self.q[self.rear], Some(ele));
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
        LimitedQueueIterator {
            lq: self,
            front_idx: 0,
            back_idx: self.sz,
        }
    }

    /// Returns a mutable iterator over the queue.
    ///
    /// # Example
    ///
    /// ```
    /// use limited_queue::LimitedQueue;
    ///
    /// let mut q = LimitedQueue::with_capacity(3);
    /// q.push(1);
    /// q.push(2);
    ///
    /// for element in q.iter_mut() {
    ///     *element *= 2;
    /// }
    ///
    /// assert_eq!(q.pop(), Some(2));
    /// assert_eq!(q.pop(), Some(4));
    /// assert_eq!(q.pop(), None);
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> LimitedQueueIteratorMut<T> {
        let len = self.sz;
        let cap = self.q.capacity();
        let front = self.front;
        let q_ptr = self.q.as_mut_ptr(); // get raw pointer to Vec's buffer
        LimitedQueueIteratorMut {
            q_ptr,
            front,
            capacity: cap,
            front_idx: 0,
            back_idx: len,
            _marker: PhantomData, // PhantomData for 'a lifetime
        }
    }

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

    /// Pop the first element from queue,
    /// will replace the element in queue
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
            ret
        }
    }
}

impl<T> std::ops::Index<usize> for LimitedQueue<T> {
    type Output = T;

    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        let real_idx = self.indexing(idx);
        self.q[real_idx].as_ref().unwrap()
    }
}

impl<T> std::ops::IndexMut<usize> for LimitedQueue<T> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        let real_idx = self.indexing(idx);
        self.q[real_idx].as_mut().unwrap()
    }
}

pub struct LimitedQueueIterator<'a, T> {
    lq: &'a LimitedQueue<T>,
    front_idx: usize,
    back_idx: usize,
}

impl<'a, T> Iterator for LimitedQueueIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front_idx == self.back_idx {
            None // the end of iteration
        } else {
            let cur_idx = self.front_idx;
            self.front_idx += 1;
            Some(&self.lq[cur_idx])
        }
    }
}

impl<'a, T> DoubleEndedIterator for LimitedQueueIterator<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front_idx == self.back_idx {
            None // the end of iteration
        } else {
            self.back_idx -= 1;
            let cur_idx = self.back_idx;
            Some(&self.lq[cur_idx])
        }
    }
}

/// A mutable, double-ended iterator over a `LimitedQueue`.
pub struct LimitedQueueIteratorMut<'a, T> {
    q_ptr: *mut Option<T>, // raw pointer to the Vec's data
    front: usize,          // internal Vec's front index
    capacity: usize,
    front_idx: usize,                // logical queue index (0..sz)
    back_idx: usize,                 // logical queue index (0..sz)
    _marker: PhantomData<&'a mut T>, // marker for 'a lifetime
}

impl<'a, T> Iterator for LimitedQueueIteratorMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front_idx == self.back_idx {
            None
        } else {
            let cur_idx = self.front_idx;
            self.front_idx += 1;
            let real_idx = (cur_idx + self.front) % self.capacity;

            unsafe {
                // get pointer to the real_idx-th element in Vec
                let elem_ptr = self.q_ptr.add(real_idx);

                // convert raw pointer back to an 'a lifetime mutable reference
                // this is safe because:
                // 1. _marker ensures 'a is the lifetime of LimitedQueueMutIterator
                // 2. we ensure real_idx is within self.q.capacity()
                // 3. cur_idx < self.sz, so this position must be Some(T)
                let opt_ref = &mut *elem_ptr;

                // .unwrap() is safe because
                // logical index (cur_idx) < self.sz,
                // which means self.q[real_idx] must be Some(T)
                Some(opt_ref.as_mut().unwrap())
            }
        }
    }
}

impl<'a, T> DoubleEndedIterator for LimitedQueueIteratorMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front_idx == self.back_idx {
            None
        } else {
            // move index back
            self.back_idx -= 1;
            let cur_idx = self.back_idx;
            let real_idx = (cur_idx + self.front) % self.capacity;

            unsafe {
                let elem_ptr = self.q_ptr.add(real_idx);
                let opt_ref = &mut *elem_ptr;

                // .unwrap() is also safe here
                Some(opt_ref.as_mut().unwrap())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic;

    use rand::Rng;

    use crate::LimitedQueue;

    // Helper struct that doesn't implement Default
    #[derive(Debug, PartialEq, Eq, Clone)]
    struct NoDefault(i32);

    #[test]
    fn test_pop_no_default() {
        let mut q = LimitedQueue::with_capacity(2);
        q.push(NoDefault(1));
        q.push(NoDefault(2));
        assert_eq!(q.push(NoDefault(3)), Some(NoDefault(1)));
        assert_eq!(q.peek(), Some(&NoDefault(2)));
        assert_eq!(q.pop(), Some(NoDefault(2)));
        assert_eq!(q.pop(), Some(NoDefault(3)));
        assert_eq!(q.pop(), None);
    }

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

        // test DoubleEndedIterator
        for (n, element) in q.iter().rev().enumerate() {
            assert_eq!(element.clone(), q[4 - n]); // 5, 6, 7, 8, 9
            assert_eq!(element.clone(), 9 - n); // 5, 6, 7, 8, 9
        }
    }

    #[test]
    fn test_iter_mut() {
        let mut q = LimitedQueue::with_capacity(5);
        for i in 0..5 {
            q.push(i); // 0, 1, 2, 3, 4
        }

        for element in q.iter_mut() {
            *element += 10;
        }

        // Queue should now contain 10, 11, 12, 13, 14
        for (n, element) in q.iter().enumerate() {
            assert_eq!(*element, n + 10);
        }

        // Test push overflow with modified data
        assert_eq!(q.push(99), Some(10)); // pops 10
        assert_eq!(q.peek(), Some(&11));
    }

    #[test]
    fn test_double_ended_iter_mut() {
        let mut q = LimitedQueue::with_capacity(5);
        for i in 0..5 {
            q.push(i); // 0, 1, 2, 3, 4
        }

        // Modify from both ends
        let mut iter_mut = q.iter_mut();
        *iter_mut.next().unwrap() = 100; // 0 -> 100
        *iter_mut.next_back().unwrap() = 400; // 4 -> 400
        *iter_mut.next().unwrap() = 200; // 1 -> 200
        *iter_mut.next_back().unwrap() = 300; // 3 -> 300
                                              // 2 is untouched

        // Check final state
        let mut iter = q.iter();
        assert_eq!(iter.next(), Some(&100));
        assert_eq!(iter.next(), Some(&200));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&300));
        assert_eq!(iter.next(), Some(&400));
        assert_eq!(iter.next(), None);
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
        LimitedQueue::<i32>::with_capacity(0)[0];
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

        // Test clear on empty
        q.clear();
        assert_eq!(q.len(), 0);
        assert!(q.is_empty());

        // Test clear on partially full
        for _ in 0..3 {
            q.push(1);
        }
        assert_eq!(q.len(), 3);
        q.clear();
        assert_eq!(q.len(), 0);
        assert_eq!(q.peek(), None);
        assert!(q.is_empty());

        // Test clear on full
        for i in 0..10 {
            q.push(i);
        }
        assert!(q.is_full());
        q.clear();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert_eq!(q.peek(), None);

        // Test functionality after clear
        assert_eq!(q.push(100), None);
        assert_eq!(q.len(), 1);
        assert_eq!(q.peek(), Some(&100));
        assert_eq!(q.pop(), Some(100));
        assert_eq!(q.len(), 0);
        assert!(q.is_empty());
    }

    #[test]
    fn test_capacity_one() {
        let mut q = LimitedQueue::with_capacity(1);
        assert!(q.is_empty());
        assert_eq!(q.push(1), None);
        assert!(q.is_full());
        assert_eq!(q.peek(), Some(&1));
        assert_eq!(q.push(2), Some(1)); // Overwrite
        assert!(q.is_full());
        assert_eq!(q.peek(), Some(&2));
        assert_eq!(q.pop(), Some(2));
        assert!(q.is_empty());
        assert_eq!(q.peek(), None);
        assert_eq!(q.pop(), None);
        assert_eq!(q.push(3), None);
        assert_eq!(q.len(), 1);
        assert_eq!(q.peek(), Some(&3));
    }

    #[test]
    #[should_panic]
    fn test_capacity_zero_push_panic() {
        let mut q = LimitedQueue::<i32>::with_capacity(0);
        q.push(1); // This should panic
    }

    #[test]
    fn test_get_method() {
        let mut q = LimitedQueue::with_capacity(3);
        assert_eq!(q.get(0), None); // Empty
        assert_eq!(q.get(1), None);

        q.push(10); // 10
        q.push(20); // 10, 20
        assert_eq!(q.get(0), Some(&10));
        assert_eq!(q.get(1), Some(&20));
        assert_eq!(q.get(2), None); // Out of bounds (len)

        q.push(30); // 10, 20, 30 (full)
        assert_eq!(q.get(2), Some(&30));
        assert_eq!(q.get(3), None);

        q.push(40); // 20, 30, 40 (wrapped)
        assert_eq!(q.get(0), Some(&20));
        assert_eq!(q.get(1), Some(&30));
        assert_eq!(q.get(2), Some(&40));
        assert_eq!(q.get(3), None); // Out of bounds (len)
    }

    #[test]
    fn test_iter_empty() {
        let mut q = LimitedQueue::<i32>::with_capacity(5);
        assert_eq!(q.iter().next(), None);
        assert_eq!(q.iter().next_back(), None);
        assert_eq!(q.iter_mut().next(), None);
        assert_eq!(q.iter_mut().next_back(), None);
    }

    #[test]
    fn test_iter_mixed() {
        let mut q = LimitedQueue::with_capacity(5);
        for i in 0..5 {
            q.push(i); // 0, 1, 2, 3, 4
        }

        let mut iter = q.iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next_back(), Some(&4));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next_back(), Some(&3));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_index_mut() {
        let mut q = LimitedQueue::with_capacity(3);
        q.push(1);
        q.push(2); // q = [1, 2]

        q[0] = 100; // Test IndexMut
        q[1] = 200;

        assert_eq!(q.get(0), Some(&100));
        assert_eq!(q.get(1), Some(&200));
    }

    #[test]
    fn test_debug_format() {
        let mut q = LimitedQueue::with_capacity(3);
        q.push(1);
        q.push(2);

        // Test whether debug works as expected
        let formatted = format!("{:?}", q);
        assert!(formatted.contains("LimitedQueue"));
        assert!(formatted.contains("Some(1)"));
        assert!(formatted.contains("Some(2)"));
    }

    #[test]
    #[should_panic(expected = "Invalid subscription index: 1")]
    fn test_indexing_panic_simple() {
        let mut q = LimitedQueue::with_capacity(3);
        q.push(1); // sz = 1
        let _ = q[1]; // Accessing q[1] when sz=1 should panic
    }
}
