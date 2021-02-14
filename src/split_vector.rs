use std::sync::Arc;
use std::slice;
use std::ops;
use std::default::Default;


pub struct SplitVectorChunk<'a, T> {
    main_memory: Arc<Vec<T>>,
    chunk: &'a mut [T]
}

impl<'a, T> SplitVectorChunk<'a, T> {
    pub fn len(&self) -> usize {
        self.chunk.len()
    }

    pub fn last(&self) -> Option<&T> {
        self.chunk.last()
    }

    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.chunk.last_mut()
    }

    pub fn raw_chunk_mut(&mut self)  -> &mut [T] {
        self.chunk
    }
}

impl<'a, T, I> ops::Index<I> for SplitVectorChunk<'a, T> 
where I: std::slice::SliceIndex<[T]> {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.chunk[index]
    }
} 

impl<'a, T, I> ops::IndexMut<I> for SplitVectorChunk<'a, T> 
where I: std::slice::SliceIndex<[T]> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.chunk[index]
    }
} 

pub struct SplitVector<T>(Arc<Vec<T>>);

impl<T: Default> SplitVector<T> {
    pub fn with_size(size: usize) -> Self {
        Self(Arc::new((0..size).map(|_| T::default()).collect::<Vec<_>>()))
    }

    pub fn with_vec(vec: Vec<T>) -> Self {
        Self(Arc::new(vec))
    }

    /**
     * Given an array of offsets, splits its contained vector into chunks based on those offsets.
     * For example, these offsets:
     *      [ a, b, c, d ]
     * Will lead to three chunks, [a..b], [b..c] and [c..d]
     * The offsets must be strictly increasing and within the range of this vector.
     */
    pub fn chunk<'a, 'b>(&'a mut self, offsets: &[usize]) -> Option<Vec<SplitVectorChunk<'b, T>>> {
        // ensure strictly ascending offsets within range
        if *offsets.last()? > self.0.len() {
            return None
        }

        let vector_start = Arc::get_mut(&mut self.0)?.as_mut_ptr();
        let mut chunks = Vec::with_capacity(offsets.len() + 1);

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

    /**
     * Similar to the chunk method, but always chunks the entire array.
     * For example, consider this vector: [ a, b, c, d, e, f, g ]
     * and this vector of offsets: [ 0, 4 ].
     * Using chunk will return these chunks: [ a, b, c, d ]
     * Using chunk_all will return these chunks: [ a, b, c, d ], [ e, f, g ]
     */
    pub fn chunk_all<'a, 'b>(&'a mut self, mut offsets: Vec<usize>) -> Option<Vec<SplitVectorChunk<'b, T>>> {
        offsets.push(self.0.len());
        self.chunk(&offsets[..])
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /**
     * Attempt to extract the inner vector, assuming no other references are held to it.  The vector
     * is returned and replaced with an empty vector.
     */
    pub fn extract(&mut self) -> Option<Vec<T>> {
        Some(std::mem::replace(Arc::get_mut(&mut self.0)?, Vec::new()))
    }

    pub fn take_vec(self) -> Option<Vec<T>> {
        Arc::try_unwrap(self.0).ok()
    }

    pub fn view_mut(&mut self) -> Option<&mut [T]> {
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
            let mut chunks = sv.chunk(&vec![0, 2, 4, 5]).unwrap();
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