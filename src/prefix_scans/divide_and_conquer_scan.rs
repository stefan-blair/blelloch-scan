use crate::prefix_scans::{Scanner, ScanError};
use crate::prefix_scans::helper_functions;
use crate::util::split_vector;
use crate::util::ranged_vector;


impl Scanner {
    /**
     * This algorithm divides the given dataset into `self.num_threads()` chunks.  Each chunk has its prefix sum
     * independently calculated by its corresponding thread.  Then, the final elements (the total sum) of each of
     * those chunks are prefix-summed, and act as the "carry" from that chunk to the next.
     *              +------------+------------+------------+------------+
     *              |  thread 0  |  thread 1  |  thread 2  |  thread 3  |
     *              +------------+------------+------------+------------+
     *                  C_0            C_1          C_2           C3
     *                                 C_0       C_0 + C_1   C_0 + C_1 + C_2
     * The first chunk is completed, and the last chunks need to have their carry-ins added to each element.
     * So, those chunks are divided amongst the threads:
     *              +------------+------------+------------+------------+
     *              |  thread 0  |  thread 1  |  thread 2  |  thread 3  |
     *              +------------+---------+--+------+-----+---+--------+
     *                           | thrd 0  | thrd 1  | thrd 2  | thrd 3 |
     *                           +---------+---------+---------+--------+
     * The threads must be careful to add the right carries to the right portions of their chunk.
     */
    pub fn divide_and_conquer_scan(&mut self, mut vec: Vec<u64>) -> Result<Vec<u64>, ScanError> {
        // partition the vector into smaller, more cache friendly sized chunks, to operate on
        for cache_chunk_start in (0..vec.len()).step_by(self.cache_chunk_length) {
            // the length of the current cache chunk.  this is either just the size of a cache chunk, or the remaining less-than cache chunk number of elements
            let current_length = std::cmp::min(self.cache_chunk_length, vec.len() - cache_chunk_start);

            // split up the current cache-chunk into smaller thread-chunks, for each thread to calculate the local prefix scan of independently
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

            // create a ranged vector for storing which carry should be used in which ranges
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

            // extract the vector back out of the SplitVector.  fails if a thread failed to release its refcount
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
            .divide_and_conquer_scan(list)
            .unwrap();
        assert_eq!(baseline, dac);
    }
}
