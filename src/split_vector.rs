use std::sync::Arc;
use std::slice;
use std::ops;


pub struct SplitVectorChunk<'a> {
    main_memory: Arc<Vec<u64>>,
    chunk: &'a mut [u64]
}

impl<'a> SplitVectorChunk<'a> {
    pub fn len(&self) -> usize {
        self.chunk.len()
    }

    pub fn last(&self) -> Option<&u64> {
        self.chunk.last()
    }

    pub fn last_mut(&mut self) -> Option<&mut u64> {
        self.chunk.last_mut()
    }
}

impl<'a> ops::Index<usize> for SplitVectorChunk<'a> {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.chunk[index]
    }
} 

impl<'a> ops::IndexMut<usize> for SplitVectorChunk<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.chunk[index]
    }
}

pub struct SplitVector(Arc<Vec<u64>>);

impl SplitVector {
    pub fn with_size(size: usize) -> Self {
        Self(Arc::new(vec![0; size]))
    }

    pub fn with_vec(vec: Vec<u64>) -> Self {
        Self(Arc::new(vec))
    }

    pub fn chunk<'a, 'b>(&'a mut self, mut offsets: Vec<usize>) -> Option<Vec<SplitVectorChunk<'b>>> {
        // ensure strictly ascending offsets within range
        if *offsets.last()? >= self.0.len() {
            return None
        }

        let vector_start = Arc::get_mut(&mut self.0)?.as_mut_ptr();
        let mut chunks = Vec::with_capacity(offsets.len() + 1);

        offsets.push(self.0.len());
        for i in 0..(offsets.len() - 1) {
            if offsets[i] >= offsets[i + 1] {
                return None
            }

            unsafe {
                chunks.push(SplitVectorChunk {
                    main_memory: self.0.clone(),
                    chunk: slice::from_raw_parts_mut(vector_start.add(offsets[i]), offsets[i + 1] - offsets[i])
                })
            }
        }

        Some(chunks)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /**
     * Attempt to extract the inner vector, assuming no other references are held to it.  The vector
     * is returned and replaced with an empty vector.
     */
    pub fn extract(&mut self) -> Option<Vec<u64>> {
        Some(std::mem::replace(Arc::get_mut(&mut self.0)?, Vec::new()))
    }

    pub fn take_vec(self) -> Option<Vec<u64>> {
        Arc::try_unwrap(self.0).ok()
    }

    pub fn view_mut(&mut self) -> Option<&mut [u64]> {
        Arc::get_mut(&mut self.0).map(|x| &mut x[..])
    }

    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::split_vector;

    #[test]
    fn basic_test() {
        let mut sv = split_vector::SplitVector::with_size(5);
        sv.view_mut().unwrap()[2] = 5;
        sv.view_mut().unwrap()[3] = 7;

        println!("original vector: {:?}", sv.view_mut());

        println!("pre chunked ref count == {:?}", sv.ref_count());
        {
            let mut chunks = sv.chunk(vec![0, 2, 4]).unwrap();
            println!("post chunked ref count == {:?}", sv.ref_count());
            for chunk in chunks.iter_mut() {
                for i in 0..chunk.chunk.len() {
                    chunk.chunk[i] += 1;
                }
            }
        }

        println!("modified vector: {:?}", sv.view_mut());
    }
}