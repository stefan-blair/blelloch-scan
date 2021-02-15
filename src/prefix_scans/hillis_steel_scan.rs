use std::sync::Arc;

use crate::prefix_scans::{Scanner, ScanError};
use crate::prefix_scans::helper_functions;
use crate::util::split_vector;


impl Scanner {
    pub fn hillis_steel_scan(&mut self, vec: Vec<u64>) -> Result<Vec<u64>, ScanError> {
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
        let mut data = Arc::new(vec);
        let mut workspace = split_vector::SplitVector::with_size(data.len());
    
        let mut step = 1;
        while step < data.len() {    
            let operation_count = data.len() - step;
            let ranges = Arc::new(helper_functions::chunk_ranges(operation_count, self.num_threads()));
            // TODO: make sure it doesn't fail when the chunks are not perfectly split up (might have idle threads)
            let split_ranges = ranges.iter().map(|i| *i + step).collect::<Vec<_>>();
            let chunks = workspace.chunk(&split_ranges).unwrap();
    
            // broadcast current iteration
            let msgs = chunks.into_iter().map(|chunk| (data.clone(), chunk, ranges.clone(), step)).collect::<Vec<_>>();
            self.thread_pool.sendall(msgs, do_step).gather().map_err(|_| ScanError::FailedThreadInGather)?;
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
        }
    
        Arc::try_unwrap(data).map_err(|_| ScanError::BrokenThreadLocking)
    }
}

#[cfg(test)]
mod test {
    use crate::prefix_scans;

    #[test]
    fn small_test() {
        let count = 12;
        let list = (0..count).collect::<Vec<_>>();

        let baseline = prefix_scans::baseline::sequential_scan_no_simd(list.clone(), |a, b| a + b).unwrap();
        let hillis_steel = prefix_scans::Scanner::new()
            .with_threads(4)
            .hillis_steel_scan(list)
            .unwrap();
        assert_eq!(baseline, hillis_steel);
    }
}