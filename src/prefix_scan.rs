use std::sync::Arc;
use std::time::{Instant};
use packed_simd;
use packed_simd::shuffle;

use crate::thread_pool;
use crate::split_vector;

pub use helper_functions::quicksum_simd;

/**
 * Move the helper functions into a separate file
 * Implement a ranged vector to help with keeping track of carries on different sized chunks for the divide and conquer
 * Instead of having a `if simd` check in each function, have the Scanner provide a sequential function that uses its internal simd flag to determine which helper function to call
 * Add a lot more comments, tests and benches
 */


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

    
    pub fn sequential_scan_simd(mut data: &mut [u64]) -> Result<(), ScanError> {
        helper_functions::prefix_scan_simd(data);
        Ok(())
    }
}

mod helper_functions {
    use super::*;

    /**
     * Performs an in-place prefix scan with addition, using simd operations.  This may not be the best implementation,
     * but it performs fairly well.
     */
    pub fn prefix_scan_simd(data: &mut [u64]) {
        let mask_1 = packed_simd::u64x8::new(0, !0, !0, !0, !0, !0, !0, !0);
        let mask_2 = packed_simd::u64x8::new(0, 0, !0, !0, !0, !0, !0, !0);
        let mask_3 = packed_simd::u64x8::new(0, 0, 0, 0, !0, !0, !0, !0);

        let mut acc = 0;
        let simd_len = (data.len() / 8) * 8;
        for i in (0..simd_len).step_by(8) {
            /*
            * Vectorize and add the acc to the next chunk in the form of a simd, so that no memory writes are needed.
            * The acc can be kept in a register instead, and moved to a simd register for addition, which is a lot faster.
            */
            let a = packed_simd::u64x8::from_slice_unaligned(&data[i..]) + packed_simd::u64x8::new(acc, 0, 0, 0, 0, 0, 0, 0);
            let b = (shuffle![a, [7, 0,1,2,3,4,5,6]] as packed_simd::u64x8) & mask_1;

            let a = a + b;
            let b = (shuffle![a, [6,7, 0,1,2,3,4,5]] as packed_simd::u64x8) & mask_2;

            let a = a + b;
            let b = (shuffle![a, [4,5,6,7, 0,1,2,3]] as packed_simd::u64x8) & mask_3;

            let a = a + b;

            acc = a.extract(7);
            a.write_to_slice_unaligned(&mut data[i..]);
        }

        for i in simd_len..data.len() {
            if i > 0 {
                data[i] += data[i - 1];
            }
        }
    }

    pub fn quicksum_simd(data: &[u64]) -> u64 {
        let simd_len = (data.len() / 8) * 8;
        let mut acc = packed_simd::u64x8::splat(0);
        for i in (0..simd_len).step_by(8) {
            let a = packed_simd::u64x8::from_slice_unaligned(&data[i..]); 
            acc = acc + a;
        }

        acc.wrapping_sum() + (&data[simd_len..data.len()]).iter().sum::<u64>()
    }

    /**
     * Returns chunks.  For example, dividing 100 into 4 chunks would yield
     * [0, 25, 50, 75]
     * Returns the start of each chunk
     */
    pub fn chunk_ranges(len: usize, num_chunks: usize) -> Vec<usize> {
        let chunk_size = len / num_chunks;
        let stragglers = len % num_chunks;
        // if there are any extra elements that dont fit all into one chunk, distribute them amongst the other chunks, from the beginning
        let large_ranges = (0..stragglers).map(|i| i * (chunk_size + 1));
        // the smaller chunks at the end that dont have any stragglers
        let small_ranges = (stragglers..num_chunks).map(|i| (i * chunk_size + stragglers));
        large_ranges.chain(small_ranges).collect()
    }
}

pub struct Scanner {
    simd_on: bool,
    sequential_length: usize,
    cache_chunk_length: usize,
    thread_pool: thread_pool::ThreadPool
}

impl Scanner {
    pub fn new() -> Self {
        let single_pool = thread_pool::ThreadPool::new(1);
        Self { simd_on: true, sequential_length: 8192, cache_chunk_length: 262144, thread_pool: single_pool }
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

    pub fn parallel_quicksum_simd(&mut self, data: &[u64]) -> u64 {
        let mut ranges = helper_functions::chunk_ranges(data.len(), self.num_threads());
        ranges.push(data.len());

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
    
    pub fn blelloch_scan<T: Default + Send + Sync + 'static>(&mut self, v: Vec<T>, func: fn(&T, &T) -> T) -> Result<Vec<T>, ScanError> {
        let mut result_vec = split_vector::SplitVector::with_vec(v);
    
        /**
         * This function, given the current step size of the pyramid, total width and number of threads, returns
         * which ranges the thread's chunks should be.  Used in both the up and down sweep loops for allocating
         * work to threads. 
         */
        fn pyramid_ranges_for(step: usize, vec_len: usize, num_threads: usize, sequential_length: usize) -> Vec<usize> {
            // total number of operands
            let num_operands = vec_len / step;
            // the total number of operations to perform this step
            let mut num_operations = num_operands / 2;
            // if there is an extra operand, and left over values, they can be combined in another operation
            if num_operands % 2 == 1 && vec_len % (step * 2) > 0 {
                num_operations += 1;
            }
    
            /*
             * The sequential_length parameter specifies a point after which everything should be sequential, because the overhead of
             * deploying to separate threads is not worth it anymore.
             */
            if num_operations < sequential_length {
                return vec![step - 1];    
            }
    
            let operation_ranges = if num_operations > num_threads {
                // because we care about distributing operations, chunk those first if theres more than one per thread
                helper_functions::chunk_ranges(num_operations, num_threads)
            } else {
                // otherwise, just assign one operation per thread
                (0..num_operations).collect::<Vec<_>>()
            };
            // get the ranges for each chunk by converting its operations to start indexes
            operation_ranges.into_iter().map(|chunk_start| chunk_start * step * 2 + step - 1).collect::<Vec<_>>()
        }
    
        // an iterator over the steps up the pyramid (1 2 4 8 ...)
        let steps = (0..((result_vec.len() as f64).log2().ceil() as usize)).map(|i| 1 << i);
    
        /*
         * First, we build up the pyramid of sections for which we know the prefix scans
         */
        for step in steps.clone() {
            // split the vector into chunks based on the pyramid ranges for the current step
            let ranges = pyramid_ranges_for(step, result_vec.len(), self.num_threads(), self.sequential_length);
            let chunks = result_vec.chunk_all(ranges).ok_or(ScanError::InvalidChunking)?.into_iter().map(|i| (step, i, func)).collect::<Vec<_>>();
            // distribute the chunks and await results
            self.thread_pool.sendall(chunks, |_, (step, mut chunk, func): (usize, split_vector::SplitVectorChunk<T>, fn(&T, &T) -> T)| {
                /*
                 * Iterate through the chunks by step * 2, skipping every other element.  should look like
                 * a  b  c  d  ...
                 * |  ^  |  ^
                 * +--+  +--+
                 * So a -> b, c -> d, instead of a -> b -> c -> d.
                 * This is what spaces out the pyramids
                 */
                for i in (0..chunk.len()).step_by(step * 2) {
                    let pair = if i + step < chunk.len() {
                        /* 
                         * The current position [i] is the peak of the last sub pyramid.  step is the width of the current
                         * sub pyramid.  [i + step] is the position of the peak of the current sub pyramid, AND the second 
                         * sub pyramid beneath this one.  Add the two to get the peak for the current pyramid.
                         */
                        i + step
                    } else if i < chunk.len() - 1 {
                        /*
                         * If theres less than step amount of extra at the end, round down to the end.
                         */
                        chunk.len() - 1
                    } else {
                        continue
                    };
    
                    let result = func(&chunk[i], &chunk[pair]);
                    chunk[pair] = result;
                }
            }).gather().map_err(|_| ScanError::FailedThreadInGather)?;
        }
    
        /*
         * Next, convert the pyramid such that each section's peak has the sum of all elements that came before the section.  The topmost peak
         * should therefore be 0
         */
        let len = result_vec.len();
        result_vec.view_mut().ok_or(ScanError::BrokenThreadLocking)?[len - 1] = T::default();
    
        /*
         * Iterate back down the pyramid, and fix each pyramid's peak to be the sum of all previous elements.  Do this by taking the left 
         * sub pyramid's peak, swapping with current peak (same elements came before left pyramid as current pyramid), and set right 
         * sub pyramid's peak to the sum of both.
         */
        for step in steps.clone().rev() {
            let ranges = pyramid_ranges_for(step, result_vec.len(), self.num_threads(), self.sequential_length);
            let chunks = result_vec.chunk_all(ranges).ok_or(ScanError::InvalidChunking)?.into_iter().map(|i| (step, i, func)).collect::<Vec<_>>();
            self.thread_pool.sendall(chunks, |_, (step, mut chunk, func): (usize, split_vector::SplitVectorChunk<T>, fn(&T, &T) -> T)| {
                for i in (0..chunk.len()).step_by(step * 2) {
                    let pair = if i + step < chunk.len() {
                        i + step
                    } else if i < chunk.len() - 1 {
                        chunk.len() - 1
                    } else {
                        continue;
                    };
    
                    // Distribute the results back down the pyramid
                    let result = func(&chunk[i], &chunk[pair]);
                    chunk[i] = std::mem::replace(&mut chunk[pair], result);
                }
            }).gather().map_err(|_| ScanError::FailedThreadInGather)?;
        }
    
        result_vec.extract().ok_or(ScanError::BrokenThreadLocking)
    }
    
    pub fn hillis_steel_scan(&mut self, vec: Vec<u64>) -> Option<Vec<u64>> {
        // individual step function
        let do_step = |(index, _), (data, mut chunk, ranges, step): (Arc<Vec<u64>>, split_vector::SplitVectorChunk<u64>, Arc<Vec<usize>>, usize)| {
            let start = ranges[index];
            // iterate over the current chunk
            for i in 0..chunk.len() {
                // performing scan operation, in this case, addition
                chunk[i] = data[start + i] + data[start + i + step];
            }
        };
    
        // allocation
        let total_elapsed = Instant::now();
        let now = Instant::now();
        let mut data = Arc::new(vec);
        let mut workspace = split_vector::SplitVector::with_size(data.len());
        println!("initializing variables: {:?}", now.elapsed().as_nanos());
    
        let now = Instant::now();
        println!("initializing thread pool: {:?}", now.elapsed().as_nanos());
    
        let mut step = 1;
        while step < data.len() {
            let now = Instant::now();
    
            let operation_count = data.len() - step;
            let ranges = Arc::new(helper_functions::chunk_ranges(operation_count, self.num_threads()));
            // TODO: make sure it doesn't fail when the chunks are not perfectly split up (might have idle threads)
            let split_ranges = ranges.iter().map(|i| *i + step).collect::<Vec<_>>();
            let chunks = workspace.chunk_all(split_ranges).unwrap();
    
            // broadcast current iteration
            let msgs = chunks.into_iter().map(|chunk| (data.clone(), chunk, ranges.clone(), step)).collect::<Vec<_>>();
            self.thread_pool.sendall(msgs, do_step).gather().ok()?;
            // try where the vectors that are filled are created here, sent with the broadcast to each thread, and then received back
            // rather than 0 it out, could just fill with garbage, but that would be unsafe
    
            let mut result = workspace.extract().unwrap();
            // this needs to be sped up
            for i in 0..step {
                result[i] = data[i];
            }
            let tmp = Arc::try_unwrap(data).unwrap();
            data = Arc::new(result);
            workspace = split_vector::SplitVector::with_vec(tmp);
    
            step <<= 1;
     
            println!("iteration with step {:?}: {:?}", step, now.elapsed().as_nanos());
        }
    
        Arc::try_unwrap(data).ok()
    }
    
    /**
     * Optimizations:  Use SIMD for addition step   (experiment with it, shuffling)
     * Instead of partitioning by thread, do a smaller partition that can fit in each thread's cache
     */
    pub fn divide_and_conquer_scan(&mut self, mut vec: Vec<u64>) -> Option<Vec<u64>> {
        let sequential_scan = |_, (mut chunk, simd): (split_vector::SplitVectorChunk<u64>, bool)| -> u64 {
            if simd {
                helper_functions::prefix_scan_simd(chunk.raw_chunk_mut());
                *chunk.last().unwrap()
            } else {
                let mut acc = 0;
                for i in 0..chunk.len() {
                    acc += chunk[i];
                    chunk[i] = acc;
                }
                acc    
            }
        };
    
        let second_sweep = |_, (mut chunk, chunk_start, ranges, carries): (split_vector::SplitVectorChunk<u64>, usize, Vec<usize>, Vec<u64>)| {
            // given a slice, and a carry, add the carry to each element in the chunk using simd instructions
            let distribute_carry = |chunk: &mut [u64], single_carry| {
                let carry = packed_simd::u64x8::splat(single_carry);
                // round the length to the nearest 8
                let multiple_length = (chunk.len() / 8) * 8;
                for i in (0..multiple_length).step_by(8) {
                    unsafe {
                        let mut quad = packed_simd::u64x8::from_slice_unaligned_unchecked(&chunk[i..]);
                        quad += carry;
                        quad.write_to_slice_unaligned_unchecked(&mut chunk[i..]);
                    }
                }
                // fill in the last few elements
                for i in multiple_length..chunk.len() {
                    chunk[i] += single_carry;
                }    
            };
    
            // these chunks are smaller than the first sweep chunks, so there can be at most two different carry ranges
            // find which carry's range we are in first
            let ranges_index = ranges.iter().enumerate().find(|(_, x)| chunk_start < **x).unwrap().0 - 1;
            
            let range_end = std::cmp::min(chunk.len(), ranges[ranges_index + 1] - chunk_start);
            distribute_carry(&mut chunk[0..range_end], carries[ranges_index]);
            if range_end < chunk.len() {
                distribute_carry(&mut chunk[range_end..], carries[ranges_index + 1]);
            }
        };

        // calculate chunks
        for cache_chunk_start in (0..vec.len()).step_by(self.cache_chunk_length) {
            // stores the length of the current cache chunk.  this is either just the size of a cache chunk, or the remaining less-than cache chunk number of elements
            let current_length = std::cmp::min(self.cache_chunk_length, vec.len() - cache_chunk_start);

            let mut first_ranges = helper_functions::chunk_ranges(current_length, self.num_threads());
            first_ranges.push(current_length);
            let mut data = split_vector::SplitVector::with_vec(vec);
            let chunks = data.chunk(&first_ranges.clone().into_iter().map(|x| x + cache_chunk_start).collect::<Vec<_>>()[..])
                        .unwrap().into_iter().map(|chunk| (chunk, self.simd_on)).collect::<Vec<_>>();
        
            // receive and accumulate the final sum for each chunk ('carry') to get the real final sums for those ranges
            let mut carries = self.thread_pool.sendall(chunks, sequential_scan).gather().ok()?;
        
            let mut acc = 0;
            for i in 0..carries.len() {
                acc += carries[i];
                carries[i] = acc;
            }
        
            // on the second sweep, the first chunk has already been calculated, and nothing is carried into it.  distribute the remaining
            // chunks, combined, over the threads
            let mut second_ranges = helper_functions::chunk_ranges(current_length - first_ranges[1], self.num_threads()).into_iter().map(|x| x + first_ranges[1]).collect::<Vec<_>>();
            second_ranges.push(current_length);
            // convert the first_ranges into the ends of their respective ranges, rather than the starts.  this is more useful for the second sweep
            let first_ranges = first_ranges.into_iter().skip(1).collect::<Vec<_>>();
            // distribute chunks and carries to add to the chunks
            let chunks = data.chunk(&second_ranges.clone().into_iter().map(|x| x + cache_chunk_start).collect::<Vec<_>>()[..]).unwrap()
                .into_iter().enumerate()
                .map(|(i, chunk)| (chunk, second_ranges[i], first_ranges.clone(), carries.clone())).collect::<Vec<_>>();
            self.thread_pool.sendall(chunks, second_sweep).gather().ok()?;

            vec = data.extract()?;
        }

        return Some(vec);
   }

   /**
     * Optimizations:  Use SIMD for addition step   (experiment with it, shuffling)
     * Instead of partitioning by thread, do a smaller partition that can fit in each thread's cache
     */
    pub fn divide_and_conquer_scan_2(&mut self, mut vec: Vec<u64>) -> Option<Vec<u64>> {
        // calculate chunks
        for cache_chunk_start in (0..vec.len()).step_by(self.cache_chunk_length) {
            // stores the length of the current cache chunk.  this is either just the size of a cache chunk, or the remaining less-than cache chunk number of elements
            let current_length = std::cmp::min(self.cache_chunk_length, vec.len() - cache_chunk_start);

            let mut ranges = helper_functions::chunk_ranges(current_length, self.num_threads());
            ranges.push(current_length);
            let ranges = ranges.into_iter()
                                .map(|x| x + cache_chunk_start)
                                .collect::<Vec<_>>();

            // receive and accumulate the final sum for each chunk ('carry') to get the real final sums for those ranges
            let borrowed_vec = Arc::new(vec);
            let mut totals = self.thread_pool
                .broadcast((borrowed_vec.clone(), ranges.clone()), |(index, _), (vec, ranges)| helper_functions::quicksum_simd(&vec[ranges[index]..ranges[index + 1]]))
                .gather().ok()?;
            vec = Arc::try_unwrap(borrowed_vec).unwrap();

            totals.pop();
            for i in 1..totals.len() {
                totals[i] += totals[i - 1]
            }

            let mut carries = vec![0];
            carries.append(&mut totals);

            let mut data = split_vector::SplitVector::with_vec(vec);
            let chunks = data.chunk(&ranges[..]).unwrap().into_iter().zip(carries.into_iter()).collect::<Vec<_>>();
            self.thread_pool.sendall(chunks, |_, (mut chunk, carry)| {
                chunk[0] += carry;
                helper_functions::prefix_scan_simd(chunk.raw_chunk_mut())
            }).gather().ok()?;
            
            vec = data.extract()?;
        }

        return Some(vec);
   }
}