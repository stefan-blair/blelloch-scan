use std::sync::Arc;

use crate::prefix_scans::{Scanner, ScanError};
use crate::prefix_scans::helper_functions;
use crate::util::split_vector;
use crate::util::ranged_vector;


impl Scanner {
    /**
     * Optimizations:  Use SIMD for addition step   (experiment with it, shuffling)
     * Instead of partitioning by thread, do a smaller partition that can fit in each thread's cache
     */
    pub fn divide_and_conquer_post_scatter_scan(&mut self, mut vec: Vec<u64>) -> Result<Vec<u64>, ScanError> {
        // calculate chunks
        for cache_chunk_start in (0..vec.len()).step_by(self.cache_chunk_length) {
            // the length of the current cache chunk.  this is either just the size of a cache chunk, or the remaining less-than cache chunk number of elements
            let current_length = std::cmp::min(self.cache_chunk_length, vec.len() - cache_chunk_start);

            let chunk_ranges = helper_functions::chunk_ranges(current_length, self.num_threads());
            let mut data = split_vector::SplitVector::with_vec(vec);
            let chunks = data.chunk(&chunk_ranges.clone().into_iter().map(|x| x + cache_chunk_start).collect::<Vec<_>>()[..]).unwrap();
        
            // receive and accumulate the final sum for each chunk ('carry') to get the real final sums for those ranges
            let mut totals = self.thread_pool.sendall(chunks, |_, mut chunk| -> u64 { 
                helper_functions::prefix_scan_simd(chunk.raw_chunk_mut());
                *chunk.last().unwrap()
            }).gather().map_err(|_| ScanError::FailedThreadInGather)?;

            // remove the last element and insert a 0 in the beginning, so that the totals are shifted down.  then prefix sum them
            totals.pop();
            let mut carries = vec![0];
            helper_functions::prefix_scan_no_simd(&mut totals[..]);
            carries.append(&mut totals);

            let carries = ranged_vector::RangedVector::new(chunk_ranges, carries);
            
            // on the second sweep, the first chunk has already been calculated, and nothing is carried into it.  distribute the remaining
            // chunks, combined, over the threads
            let ranges = helper_functions::chunk_ranges(current_length - carries.get_range(0).unwrap().end(), self.num_threads())
                .into_iter().map(|x| x + carries.get_range(0).unwrap().end())
                .collect::<Vec<_>>();
            // distribute chunks and carries to add to the chunks
            let chunks = data.chunk(&ranges.clone().into_iter().map(|x| x + cache_chunk_start).collect::<Vec<_>>()).unwrap()
                .into_iter().enumerate().map(|(i, chunk)| (chunk, ranges[i], carries.clone())).collect::<Vec<_>>();
            self.thread_pool.sendall(chunks, |_, (mut chunk, chunk_start, carries)| {
                // these chunks are smaller than the first sweep chunks, so there can be at most two different carry ranges
                // find which carry's range we are in first
                let carry_range = carries.get(chunk_start).unwrap();
                let carry_range_distance = std::cmp::min(chunk.len(), carry_range.end() - chunk_start);
                helper_functions::add_to_all_simd(*carry_range.value(), &mut chunk[0..carry_range_distance]);
                if carry_range_distance < chunk.len() {
                    helper_functions::add_to_all_simd(*carries.next_range(carry_range).unwrap().value(), &mut chunk[carry_range_distance..]);
                }
            }).gather().map_err(|_| ScanError::FailedThreadInGather)?;

            vec = data.extract().ok_or(ScanError::BrokenThreadLocking)?;
        }

        return Ok(vec);
   }

    pub fn divide_and_conquer_pre_scatter_scan(&mut self, mut vec: Vec<u64>) -> Result<Vec<u64>, ScanError> {
        // calculate chunks
        for cache_chunk_start in (0..vec.len()).step_by(self.cache_chunk_length) {
            // stores the length of the current cache chunk.  this is either just the size of a cache chunk, or the remaining less-than cache chunk number of elements
            let current_length = std::cmp::min(self.cache_chunk_length, vec.len() - cache_chunk_start);

            let ranges = helper_functions::chunk_ranges(current_length, self.num_threads());
            let ranges = ranges.into_iter()
                                .map(|x| x + cache_chunk_start)
                                .collect::<Vec<_>>();

            // receive and accumulate the final sum for each chunk ('carry') to get the real final sums for those ranges
            let borrowed_vec = Arc::new(vec);
            let mut totals = self.thread_pool
                .broadcast((borrowed_vec.clone(), ranges.clone()), |(index, _), (vec, ranges)| helper_functions::quicksum_simd(&vec[ranges[index]..ranges[index + 1]]))
                .gather().map_err(|_| ScanError::FailedThreadInGather)?;
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
            }).gather().map_err(|_| ScanError::FailedThreadInGather)?;
            
            vec = data.extract().ok_or(ScanError::BrokenThreadLocking)?;
        }

        return Ok(vec);
   }
}

#[cfg(test)]
mod test {
    use crate::prefix_scans;

    #[test]
    fn small_post_scatter_test() {
        let count = 12;
        let list = (0..count).collect::<Vec<_>>();

        let baseline = prefix_scans::baseline::sequential_scan_no_simd(list.clone(), |a, b| a + b).unwrap();
        let dac = prefix_scans::Scanner::new()
            .with_threads(4)
            .divide_and_conquer_post_scatter_scan(list)
            .unwrap();
        assert_eq!(baseline, dac);
    }

    #[test]
    fn small_pre_scatter_test() {
        let count = 12;
        let list = (0..count).collect::<Vec<_>>();

        let baseline = prefix_scans::baseline::sequential_scan_no_simd(list.clone(), |a, b| a + b).unwrap();
        let dac = prefix_scans::Scanner::new()
            .with_threads(4)
            .divide_and_conquer_pre_scatter_scan(list)
            .unwrap();
        assert_eq!(baseline, dac);
    }
}
