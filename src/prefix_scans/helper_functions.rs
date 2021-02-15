use packed_simd;
use packed_simd::shuffle;


pub fn prefix_scan_no_simd(data: &mut [u64]) {
    for i in 1..data.len() {
        data[i] += data[i - 1];
    }
}

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

pub fn add_to_all_simd(value: u64, data: &mut [u64]) {
    let value_vector = packed_simd::u64x8::splat(value);
    // round the length to the nearest 8
    let multiple_length = (data.len() / 8) * 8;
    for i in (0..multiple_length).step_by(8) {
        unsafe {
            let mut quad = packed_simd::u64x8::from_slice_unaligned_unchecked(&data[i..]);
            quad += value_vector;
            quad.write_to_slice_unaligned_unchecked(&mut data[i..]);
        }
    }
    // fill in the last few elements
    for i in multiple_length..data.len() {
        data[i] += value;
    }
}

/**
 * Returns chunks.  For example, dividing 100 into 4 chunks would yield
 * [0, 25, 50, 75, 100]
 */
pub fn chunk_ranges(len: usize, num_chunks: usize) -> Vec<usize> {
    let chunk_size = len / num_chunks;
    let stragglers = len % num_chunks;
    // if there are any extra elements that dont fit all into one chunk, distribute them amongst the other chunks, from the beginning
    let large_ranges = (0..stragglers).map(|i| i * (chunk_size + 1));
    // the smaller chunks at the end that dont have any stragglers
    let small_ranges = (stragglers..(num_chunks + 1)).map(|i| (i * chunk_size + stragglers));
    large_ranges.chain(small_ranges).collect()
}

#[cfg(test)]
mod test {
    use crate::prefix_scans;

    #[test]
    fn simd_sequential_test() {
        let count = 32;
        let mut list = (0..count).collect::<Vec<_>>();
        let baseline = list.clone();
        prefix_scans::baseline::sequential_scan_simd(&mut list[..]).unwrap();
        assert_eq!(prefix_scans::baseline::sequential_scan_no_simd(baseline, |a, b| a + b).unwrap(), list)
    }

    #[test]
    fn quicksum_test() {
        let vec = (0..35).collect::<Vec<_>>();
        assert_eq!(prefix_scans::helper_functions::quicksum_simd(&vec), vec.iter().sum());
    }
}