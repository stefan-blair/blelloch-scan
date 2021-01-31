use std::sync::Arc;
use std::time::{Instant};

use crate::thread_pool;
use crate::split_vector;


#[derive(Debug)]
pub enum ScanError {
    BrokenThreadLocking,
    FailedThreadInGather,
    InvalidChunking
}

pub struct HeadFlagVec {
    pub data: Vec<u64>,
    pub head_flags: Vec<bool>
}

impl HeadFlagVec {
    pub fn from_vec(v: &Vec<Vec<u64>>) -> HeadFlagVec {
        let data: Vec<u64> = v.iter().flatten().copied().collect();
        let mut head_flags = vec![false; data.len()];

        let mut flat_i = 0;
        for sub_v in v.iter() {
            head_flags[flat_i] = true;
            flat_i += sub_v.len();
        }

        Self {data, head_flags}
    }

    pub fn to_vec(&self) -> Option<Vec<Vec<u64>>> {
        if self.head_flags.len() > 0 && self.head_flags[0] != true {
            return None
        }

        let mut vec = Vec::new();
        for (is_head, data) in self.head_flags.iter().copied().zip(self.data.iter().copied()) {
            if is_head {
                vec.push(Vec::new())
            }

            vec.last_mut().unwrap().push(data);
        }

        Some(vec)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, index: usize) -> (u64, bool) {
        (self.data[index], self.head_flags[index])
    }
}

pub fn sequential_scan(v: &Vec<Vec<u64>>) -> Option<Vec<Vec<u64>>> {
    let mut vec = HeadFlagVec::from_vec(v);
 
    let mut scan = 0;
    for i in 0..vec.len() {
        if vec.head_flags[i] {
            scan = 0;
        }
        scan += vec.data[i];
        vec.data[i] = scan;
    }

    vec.to_vec()
}

pub fn chunk_ranges(len: usize, num_chunks: usize) -> Vec<usize> {
    let chunk_size = len / num_chunks;
    let stragglers = len % num_chunks;
    // if there are any extra elements that dont fit all into one chunk, distribute them amongst the other chunks, from the beginning
    let large_ranges = (0..stragglers).map(|i| i * (chunk_size + 1));
    // the smaller chunks at the end that dont have any stragglers
    let small_ranges = (stragglers..num_chunks).map(|i| (i * chunk_size + stragglers));
    large_ranges.chain(small_ranges).collect()
}

/**
 * Potential optimizations:
 * Make some threshold below which everything is done on a single thread (not worth parellel overhead)
 */
pub fn blelloch_scan(num_threads: usize, v: &Vec<u64>) -> Result<Vec<u64>, ScanError> {
    let mut pool = thread_pool::ThreadPool::new(num_threads);
    let mut result_vec = split_vector::SplitVector::with_vec(v.clone());

    /**
     * This function, given the current step size of the pyramid, total width and number of threads, returns
     * which ranges the thread's chunks should be.  Used in both the up and down sweep loops for allocating
     * work to threads. 
     */
    fn pyramid_ranges_for(step: usize, vec_len: usize, num_threads: usize) -> Vec<usize> {
        // total number of operands
        let num_operands = vec_len / step;
        // the total number of operations to perform this step
        let mut num_operations = num_operands / 2;
        // if there is an extra operand, and left over values, they can be combined in another operation
        if num_operands % 2 == 1 && vec_len % (step * 2) > 0 {
            num_operations += 1;
        }
        let operation_ranges = if num_operations > num_threads {
            // because we care about distributing operations, chunk those first if theres more than one per thread
            chunk_ranges(num_operations, num_threads)
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
        let ranges = pyramid_ranges_for(step, result_vec.len(), num_threads);
        let chunks = result_vec.chunk(ranges).ok_or(ScanError::InvalidChunking)?.into_iter().map(|i| (step, i)).collect::<Vec<_>>();
        // distribute the chunks and await results
        pool.sendall(chunks, |_, (step, mut chunk): (usize, split_vector::SplitVectorChunk<u64>)| {
            /*
             * Iterate through the chunks by step * 2, skipping every other element.  should look like
             * a  b  c  d  ...
             * |  ^  |  ^
             * +--+  +--+
             * So a -> b, c -> d, instead of a -> b -> c -> d.
             * This is what spaces out the pyramids
             */
            for i in (0..chunk.len()).step_by(step * 2) {
                if i + step < chunk.len() {
                    /* 
                     * The current position [i] is the peak of the last sub pyramid.  step is the width of the current
                     * sub pyramid.  [i + step] is the position of the peak of the current sub pyramid, AND the second 
                     * sub pyramid beneath this one.  Add the two to get the peak for the current pyramid.
                     */
                    chunk[i + step] += chunk[i];
                } else if i < chunk.len() - 1 {
                    /*
                     * If theres less than step amount of extra at the end, round down to the end.
                     */
                    *chunk.last_mut().unwrap() += chunk[i];
                }
            }
        }).gather().map_err(|_| ScanError::FailedThreadInGather)?;
    }

    /*
     * Next, convert the pyramid such that each section's peak has the sum of all elements that came before the section.  The topmost peak
     * should therefore be 0
     */
    let len = result_vec.len();
    result_vec.view_mut().ok_or(ScanError::BrokenThreadLocking)?[len - 1] = 0;

    /*
     * Iterate back down the pyramid, and fix each pyramid's peak to be the sum of all previous elements.  Do this by taking the left 
     * sub pyramid's peak, swapping with current peak (same elements came before left pyramid as current pyramid), and set right 
     * sub pyramid's peak to the sum of both.
     */
    for step in steps.clone().rev() {
        let ranges = pyramid_ranges_for(step, result_vec.len(), num_threads);
        let chunks = result_vec.chunk(ranges).ok_or(ScanError::InvalidChunking)?.into_iter().map(|i| (step, i)).collect::<Vec<_>>();
        pool.sendall(chunks, |_, (step, mut chunk): (usize, split_vector::SplitVectorChunk<u64>)| {
            for i in (0..chunk.len()).step_by(step * 2) {
                let pair = if i + step < chunk.len() {
                    i + step
                } else if i < chunk.len() - 1 {
                    chunk.len() - 1
                } else {
                    continue;
                };

                // Distribute the results back down the pyramid
                let tmp = chunk[pair];
                chunk[pair] += chunk[i];
                chunk[i] = tmp;               
            }
        }).gather().map_err(|_| ScanError::FailedThreadInGather)?;
    }

    result_vec.extract().ok_or(ScanError::BrokenThreadLocking)
}

pub fn hillis_steel_scan(num_threads: usize, v: &Vec<Vec<u64>>) -> Option<Vec<Vec<u64>>> {
    // individual step function
    let do_step = |(index, _), (data, _head_flags, mut chunk, ranges, step): (Arc<Vec<u64>>, _, split_vector::SplitVectorChunk<u64>, Arc<Vec<usize>>, _)| {
        let start = ranges[index];
        // iterate over the current chunk
        for i in 0..chunk.len() {
            // performing scan operation, in this case, addition
            chunk[i] = data[start + i] + data[start + i + step];
        }
    };

    // allocation
    let mut vec = HeadFlagVec::from_vec(v);

    let total_elapsed = Instant::now();
    let now = Instant::now();
    let head_flags = Arc::new(vec.head_flags);
    let mut data = Arc::new(vec.data);
    let mut workspace = split_vector::SplitVector::with_size(data.len());
    println!("initializing variables: {:?}", now.elapsed().as_nanos());

    let now = Instant::now();
    let mut pool = thread_pool::ThreadPool::new(num_threads);
    println!("initializing thread pool: {:?}", now.elapsed().as_nanos());

    let mut step = 1;
    while step < head_flags.len() {
        let now = Instant::now();

        let operation_count = data.len() - step;
        let ranges = Arc::new(chunk_ranges(operation_count, num_threads));
        // TODO: make sure it doesn't fail when the chunks are not perfectly split up (might have idle threads)
        let split_ranges = ranges.iter().map(|i| *i + step).collect::<Vec<_>>();
        let chunks = workspace.chunk(split_ranges).unwrap();

        // broadcast current iteration
        let msgs = chunks.into_iter().map(|chunk| (data.clone(), head_flags.clone(), chunk, ranges.clone(), step)).collect::<Vec<_>>();
        pool.sendall(msgs, do_step).gather().ok()?;
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

    vec.data = Arc::try_unwrap(data).unwrap();
    vec.head_flags = Arc::try_unwrap(head_flags).unwrap();

    println!("total time elapsed: {:?}", total_elapsed.elapsed().as_nanos());

    vec.to_vec()
}

pub fn divide_and_conquer_scan(num_threads: usize, v: &Vec<Vec<u64>>) -> Option<Vec<Vec<u64>>> {
    let first_sweep = |_, mut chunk: split_vector::SplitVectorChunk<u64>| -> u64 {
        let mut acc = 0;
        for i in 0..chunk.len() {
            acc += chunk[i];
            chunk[i] = acc;
        }
        acc
    };

    let second_sweep = |(index, _), (mut chunk, carries): (split_vector::SplitVectorChunk<u64>, Arc<Vec<u64>>)| {
        if index > 0 {
            for i in 0..chunk.len() {
                chunk[i] += carries[index - 1]
            }
        }
    };

    let mut vec = HeadFlagVec::from_vec(v);

    let now = Instant::now();

    let chunk_ranges = chunk_ranges(vec.len(), num_threads);
    let mut data = split_vector::SplitVector::with_vec(vec.data);
    let chunks = data.chunk(chunk_ranges.clone()).unwrap();

    // calculate chunks
    let mut pool = thread_pool::ThreadPool::new(num_threads);
    // receive and accumulate the final sum for each chunk ('left') to get the real final sums for those ranges
    let mut lefts = pool.sendall(chunks, first_sweep).gather().ok()?;

    let mut acc = 0;
    for i in 0..lefts.len() {
        acc += lefts[i];
        lefts[i] = acc;
    }
    let lefts = Arc::new(lefts);

    // distribute chunks and lefts to add to the chunks
    let chunks = data.chunk(chunk_ranges).unwrap().into_iter().map(|chunk| (chunk, lefts.clone())).collect::<Vec<_>>();
    pool.sendall(chunks, second_sweep).gather().ok()?;

    vec.data = data.extract()?;

    println!("Divide and Conquer scan: {:?}", now.elapsed().as_nanos());

    vec.to_vec()
}
