use crate::prefix_scans::{Scanner, ScanError};
use crate::prefix_scans::helper_functions;
use crate::util::split_vector;


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
        (0..num_operations + 1).collect::<Vec<_>>()
    };
    // get the ranges for each chunk by converting its operations to start indexes
    let mut ranges = operation_ranges.into_iter().map(|chunk_start| chunk_start * step * 2 + step - 1).collect::<Vec<_>>();
    // make sure the last element rounds down to the size of the vector
    *ranges.last_mut().unwrap() = vec_len;
    return ranges
}

impl Scanner {
    pub fn blelloch_scan_generic<T: Default + Send + Sync + 'static>(&mut self, v: Vec<T>, func: fn(&T, &T) -> T) -> Result<Vec<T>, ScanError> {
        let mut result_vec = split_vector::SplitVector::with_vec(v);
    
        // an iterator over the steps up the pyramid (1 2 4 8 ...)
        let steps = (0..((result_vec.len() as f64).log2().ceil() as usize)).map(|i| 1 << i);
    
        /*
         * First, we build up the pyramid of sections for which we know the total scans
         */
        for step in steps.clone() {
            // split the vector into chunks based on the pyramid ranges for the current step
            let ranges = pyramid_ranges_for(step, result_vec.len(), self.num_threads(), self.sequential_length);
            let chunks = result_vec.chunk(&ranges).ok_or(ScanError::InvalidChunking)?.into_iter().map(|i| (step, i, func)).collect::<Vec<_>>();
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
            let chunks = result_vec.chunk(&ranges).ok_or(ScanError::InvalidChunking)?.into_iter().map(|i| (step, i, func)).collect::<Vec<_>>();
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

    pub fn blelloch_scan(&mut self, v: Vec<u64>) -> Result<Vec<u64>, ScanError> {
        let mut result_vec = split_vector::SplitVector::with_vec(v);
    
        // an iterator over the steps up the pyramid (1 2 4 8 ...)
        let steps = (0..((result_vec.len() as f64).log2().ceil() as usize)).map(|i| 1 << i);
    
        /*
         * First, we build up the pyramid of sections for which we know the total scans
         */
        for step in steps.clone() {
            // split the vector into chunks based on the pyramid ranges for the current step
            let ranges = pyramid_ranges_for(step, result_vec.len(), self.num_threads(), self.sequential_length);
            let chunks = result_vec.chunk(&ranges).ok_or(ScanError::InvalidChunking)?.into_iter().map(|i| (step, i)).collect::<Vec<_>>();
            // distribute the chunks and await results
            self.thread_pool.sendall(chunks, |_, (step, mut chunk): (usize, split_vector::SplitVectorChunk<u64>)| {
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
    
                    let result = chunk[i] + chunk[pair];
                    chunk[pair] = result;
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
            let ranges = pyramid_ranges_for(step, result_vec.len(), self.num_threads(), self.sequential_length);
            let chunks = result_vec.chunk(&ranges).ok_or(ScanError::InvalidChunking)?.into_iter().map(|i| (step, i)).collect::<Vec<_>>();
            self.thread_pool.sendall(chunks, |_, (step, mut chunk): (usize, split_vector::SplitVectorChunk<u64>)| {
                for i in (0..chunk.len()).step_by(step * 2) {
                    let pair = if i + step < chunk.len() {
                        i + step
                    } else if i < chunk.len() - 1 {
                        chunk.len() - 1
                    } else {
                        continue;
                    };
    
                    // Distribute the results back down the pyramid
                    let result = chunk[i] + chunk[pair];
                    chunk[i] = std::mem::replace(&mut chunk[pair], result);
                }
            }).gather().map_err(|_| ScanError::FailedThreadInGather)?;
        }
    
        result_vec.extract().ok_or(ScanError::BrokenThreadLocking)
    }
}

#[cfg(test)]
mod test {
    use crate::prefix_scans;

    #[test]
    fn small_test() {
        let list = (0..12).collect::<Vec<_>>();

        let baseline = prefix_scans::baseline::sequential_scan_no_simd(list.clone(), |a, b| a + b).unwrap();
        let blelloch = prefix_scans::Scanner::new()
            .with_threads(4)
            .blelloch_scan(list)
            .unwrap();
        assert_eq!(baseline.split_last().unwrap().1, &blelloch[1..]);
    }

    #[test]
    fn medium_500000_test() {
        let list = (0..500000).collect::<Vec<u64>>();

        let baseline = prefix_scans::baseline::sequential_scan_no_simd(list.clone(), |a, b| a + b).unwrap();
        let blelloch = prefix_scans::Scanner::new()
            .with_threads(4)
            .blelloch_scan(list)
            .unwrap();
        assert_eq!(baseline.split_last().unwrap().1, &blelloch[1..]);
    }
}