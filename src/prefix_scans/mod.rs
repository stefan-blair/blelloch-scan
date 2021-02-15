use crate::util::thread_pool;

pub mod helper_functions;
pub mod blelloch_scan;
pub mod hillis_steel_scan;
pub mod divide_and_conquer_scan;


#[derive(Debug)]
pub enum ScanError {
    BrokenThreadLocking,
    FailedThreadInGather,
    InvalidChunking
}

pub mod baseline {
    use super::*;

    pub fn sequential_scan_no_simd<T: Default + Send + Sync + 'static>(mut vec: Vec<T>, func: fn(&T, &T) -> T) -> Result<Vec<T>, ScanError> {
        for i in 1..vec.len() {
            let (x, y) = (&vec[i - 1], &vec[i]);
            let result = func(x, y);
            vec[i] = result;
        }

        Ok(vec)
    }

    
    pub fn sequential_scan_simd(data: &mut [u64]) -> Result<(), ScanError> {
        helper_functions::prefix_scan_simd(data);
        Ok(())
    }
}

pub struct Scanner {
    simd_on: bool,
    sequential_length: usize,
    cache_chunk_length: usize,
    thread_pool: thread_pool::ThreadPool
}

/**
 * Initialization functions.
 */
impl Scanner {
    pub fn new() -> Self {
        let single_pool = thread_pool::ThreadPool::new(1);
        Self { simd_on: true, sequential_length: 0, cache_chunk_length: 262144, thread_pool: single_pool }
    }

    pub fn without_simd(mut self) -> Self {
        self.simd_on = false;
        self
    }

    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.thread_pool = thread_pool::ThreadPool::new(num_threads);
        self
    }

    pub fn with_sequential_length(mut self, sequential_length: usize) -> Self {
        self.sequential_length = sequential_length;
        self
    }

    pub fn with_cache_chunk_length(mut self, cache_chunk_length: usize) -> Self {
        self.cache_chunk_length = cache_chunk_length;
        self
    }

    pub fn num_threads(&self) -> usize {
        self.thread_pool.num_threads()
    }

    pub fn set_sequential_length(&mut self, sequential_length: usize) {
        self.sequential_length = sequential_length
    }

    pub fn set_cache_chunk_length(&mut self, cache_chunk_length: usize) {
        self.cache_chunk_length = cache_chunk_length
    }

    pub fn parallel_quicksum_simd(&mut self, data: &[u64]) -> u64 {
        let ranges = helper_functions::chunk_ranges(data.len(), self.num_threads());

        // we know that the threads will finish by the end of the function, hack around the lifetimes
        let data_len = data.len();
        let data_ptr = data.as_ptr();
        unsafe {
            let data = std::slice::from_raw_parts(data_ptr, data_len);
            self.thread_pool.broadcast((data, ranges), |(index, _), (data, ranges)| -> u64 {
                helper_functions::quicksum_simd(&data[ranges[index]..ranges[index + 1]])
            }).gather().unwrap().into_iter().sum()
        }
    }
}


#[cfg(test)]
mod test {
    use crate::prefix_scans;

    #[test]
    fn parallel_quicksum_test() {
        let vec = (0..35).collect::<Vec<_>>();
        assert_eq!(prefix_scans::Scanner::new().with_threads(4).parallel_quicksum_simd(&vec), vec.iter().sum());        
    }
}