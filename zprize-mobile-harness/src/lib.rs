use ark_bls12_377 as bls377;
use ark_ec::Group;
use ark_ff::fields::Field;
use ark_ff::BigInteger;
use ark_ff::PrimeField;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;
use ark_serialize::Compress;
use ark_std::io::Write;
use ark_std::rand::Rng;
use ark_std::{One, Zero};
use rand::RngCore;
use std::collections::VecDeque;
use std::fs::File;
use std::time::Duration;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HarnessError {
    #[error("could not serialize")]
    SerializationError(#[from] ark_serialize::SerializationError),

    #[error("could not open file")]
    FileOpenError(#[from] std::io::Error),

    #[error("failed to read at least one instance from file")]
    DeserializationError,
}

type Point = bls377::G1Affine;
type ProjectivePoint = bls377::G1Projective;
type ScalarField = bls377::Fr;
type Scalar = <ScalarField as PrimeField>::BigInt;
type Instance = (Vec<Point>, Vec<Scalar>);

fn ln_without_floats(a: usize) -> usize {
    // log2(a) * ln(2)
    (ark_std::log2(a) * 69 / 100) as usize
}

/// Optimized implementation of multi-scalar multiplication.
fn msm_bigint(bases: &[Point], bigints: &[Scalar]) -> ProjectivePoint {
    let size = ark_std::cmp::min(bases.len(), bigints.len());
    let scalars = &bigints[..size];
    let bases = &bases[..size];
    let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _)| !s.is_zero());

    let c = if size < 32 {
        3
    } else {
        ln_without_floats(size) + 2
    };

    let num_bits = ScalarField::MODULUS_BIT_SIZE as usize;
    let fr_one = ScalarField::one().into_bigint();

    let zero = ProjectivePoint::zero();
    let window_starts: Vec<_> = (0..num_bits).step_by(c).collect();

    // Each window is of size `c`.
    // We divide up the bits 0..num_bits into windows of size `c`, and
    // in parallel process each such window.
    let window_sums: Vec<_> = ark_std::cfg_into_iter!(window_starts)
        .map(|w_start| {
            let mut res = zero;
            // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
            let mut buckets = vec![zero; (1 << c) - 1];
            // This clone is cheap, because the iterator contains just a
            // pointer and an index into the original vectors.
            scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {
                if scalar == fr_one {
                    // We only process unit scalars once in the first window.
                    if w_start == 0 {
                        res += base;
                    }
                } else {
                    let mut scalar_copy = scalar;

                    // We right-shift by w_start, thus getting rid of the
                    // lower bits.
                    scalar_copy.divn(w_start as u32);

                    // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
                    let bucket_idx = scalar_copy.as_ref()[0] % (1 << c);

                    // If the bucket_idx is non-zero, we update the corresponding
                    // bucket.
                    // (Recall that `buckets` doesn't have a zero bucket.)
                    if bucket_idx != 0 {
                        buckets[(bucket_idx - 1) as usize] += base;
                    }
                }
            });

            // Compute sum_{i in 0..num_buckets} (sum_{j in i..num_buckets} bucket[j])
            // This is computed below for b buckets, using 2b curve additions.
            //
            // We could first normalize `buckets` and then use mixed-addition
            // here, but that's slower for the kinds of groups we care about
            // (Short Weierstrass curves and Twisted Edwards curves).
            // In the case of Short Weierstrass curves,
            // mixed addition saves ~4 field multiplications per addition.
            // However normalization (with the inversion batched) takes ~6
            // field multiplications per element,
            // hence batch normalization is a slowdown.

            // `running_sum` = sum_{j in i..num_buckets} bucket[j],
            // where we iterate backward from i = num_buckets to 0.
            let mut running_sum = ProjectivePoint::zero();
            buckets.into_iter().rev().for_each(|b| {
                running_sum += &b;
                res += &running_sum;
            });
            res
        })
        .collect();

    // We store the sum for the lowest window.
    let lowest = *window_sums.first().unwrap();

    // We're traversing windows from high to low.
    lowest
        + &window_sums[1..]
            .iter()
            .rev()
            .fold(zero, |mut total, sum_i| {
                total += sum_i;
                for _ in 0..c {
                    total.double_in_place();
                }
                total
            })
}

pub fn gen_random_vectors<R: RngCore>(n: usize, rng: &mut R) -> Instance {
    let num_bytes = bls377::Fr::zero().serialized_size(Compress::Yes);
    let mut points = Vec::<Point>::new();
    let mut scalars = Vec::<Scalar>::new();
    let mut bytes = vec![0; num_bytes];
    let mut scalar;
    for _i in 0..n {
        loop {
            rng.fill_bytes(&mut bytes[..]);
            scalar = bls377::Fr::from_random_bytes(&bytes);
            if scalar.is_some() {
                break;
            }
        }
        scalars.push(scalar.unwrap().into_bigint());

        let point: bls377::G1Projective = rng.gen();
        points.push(point.into());
    }
    (points, scalars)
}

pub fn gen_zero_vectors<R: RngCore>(n: usize, rng: &mut R) -> Instance {
    let num_bytes = bls377::Fr::zero().serialized_size(Compress::Yes);
    let mut points = Vec::<Point>::new();
    let mut scalars = Vec::<Scalar>::new();
    let mut bytes = vec![0; num_bytes];
    let mut scalar;
    for _i in 0..n {
        rng.fill_bytes(&mut bytes[..]);
        scalar = bls377::Fr::zero();
        scalars.push(scalar.into_bigint());

        let point: bls377::G1Projective = rng.gen();
        points.push(point.into());
    }
    (points, scalars)
}

pub fn serialize_input(
    dir: &str,
    points: &[Point],
    scalars: &[Scalar],
    append: bool,
) -> Result<(), HarnessError> {
    let points_path = format!("{}{}", dir, "/points");
    let scalars_path = format!("{}{}", dir, "/scalars");
    let (mut f1, mut f2) = if append {
        let file1 = File::options()
            .append(true)
            .create(true)
            .open(points_path)?;
        let file2 = File::options()
            .append(true)
            .create(true)
            .open(scalars_path)?;
        (file1, file2)
    } else {
        let file1 = File::create(points_path)?;
        let file2 = File::create(scalars_path)?;
        (file1, file2)
    };
    points.serialize_compressed(&mut f1)?;
    scalars.serialize_compressed(&mut f2)?;
    Ok(())
}

pub enum FileInputIteratorMode {
    Checked,
    Unchecked,
}

pub struct FileInputIterator {
    points_file: File,
    scalars_file: File,
    mode: FileInputIteratorMode,
    cached: Option<Instance>,
}

impl FileInputIterator {
    pub fn open(dir: &str) -> Result<Self, HarnessError> {
        let points_path = format!("{}{}", dir, "/points");
        let scalars_path = format!("{}{}", dir, "/scalars");

        // Try to read an instance, first in uncheck, then check serialization modes.
        let mut iter = Self {
            points_file: File::open(&points_path)?,
            scalars_file: File::open(&scalars_path)?,
            mode: FileInputIteratorMode::Unchecked,
            cached: None,
        };

        // Read a first value and see if we get a result.
        iter.cached = iter.next();
        if iter.cached.is_some() {
            return Ok(iter);
        }

        let mut iter = Self {
            points_file: File::open(&points_path)?,
            scalars_file: File::open(&scalars_path)?,
            mode: FileInputIteratorMode::Checked,
            cached: None,
        };
        iter.cached = iter.next();
        if iter.cached.is_none() {
            return Err(HarnessError::DeserializationError);
        }
        Ok(iter)
    }
}

impl Iterator for FileInputIterator {
    type Item = Instance;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cached.is_some() {
            return self.cached.take();
        }

        let points = match self.mode {
            FileInputIteratorMode::Checked => {
                Vec::<Point>::deserialize_compressed(&mut self.points_file)
            },
            FileInputIteratorMode::Unchecked => {
                Vec::<Point>::deserialize_compressed_unchecked(&mut self.points_file)
            },
        };

        let points = match points {
            Ok(x) => Some(x),
            Err(_) => None,
        }?;

        let scalars = Vec::<Scalar>::deserialize_compressed(&mut self.scalars_file);
        let scalars = match scalars {
            Ok(x) => Some(x),
            Err(_) => None,
        }?;

        Some((points, scalars))
    }
}

pub struct VectorInputIterator {
    points: VecDeque<Vec<Point>>,
    scalars: VecDeque<Vec<Scalar>>,
}

impl Iterator for VectorInputIterator {
    type Item = Instance;

    fn next(&mut self) -> Option<Self::Item> {
        let points = self.points.pop_front()?;
        let scalars = self.scalars.pop_front()?;
        Some((points, scalars))
    }
}

impl From<Instance> for VectorInputIterator {
    fn from(other: Instance) -> Self {
        Self {
            points: vec![other.0].into(),
            scalars: vec![other.1].into(),
        }
    }
}

impl From<(Vec<Vec<Point>>, Vec<Vec<Scalar>>)> for VectorInputIterator {
    fn from(other: (Vec<Vec<Point>>, Vec<Vec<Scalar>>)) -> Self {
        Self {
            points: other.0.into(),
            scalars: other.1.into(),
        }
    }
}

pub fn benchmark_msm<I>(
    output_dir: &str,
    instances: I,
    iterations: u32,
) -> Result<Vec<Duration>, HarnessError>
where
    I: Iterator<Item = Instance>,
{
    let output_path = format!("{}{}", output_dir, "/resulttimes.txt");
    let output_result_path = format!("{}{}", output_dir, "/result.txt");
    let mut output_file = File::create(output_path).expect("output file creation failed");
    let mut output_result_file =
        File::create(output_result_path).expect("output file creation failed");
    let mut result_vec = Vec::new();

    for instance in instances {
        let points = &instance.0;
        let scalars = &instance.1;

        let mut total_duration = Duration::ZERO;
        for i in 0..iterations {
            let start = Instant::now();
            let result: bls377::G1Projective = msm_bigint(&points[..], &scalars[..]);
            let time = start.elapsed();
            writeln!(&mut output_file, "iteration {}: {:?}", i + 1, time)?;
            result.serialize_uncompressed(&mut output_result_file)?;
            total_duration += time;
        }
        let mean = total_duration / iterations;
        writeln!(&mut output_file, "Mean across all iterations: {:?}", mean)?;
        println!(
            "Average time to execute MSM with {} points and {} scalars and {} iterations is: {:?}",
            points.len(),
            scalars.len(),
            iterations,
            mean
        );
        result_vec.push(mean);
    }
    Ok(result_vec)
}

/// Expose the JNI interface for android below
#[cfg(target_os = "android")]
#[allow(non_snake_case)]
pub mod android {
    extern crate jni;
    use self::jni::objects::{JClass, JString};
    use self::jni::sys::jstring;
    use self::jni::JNIEnv;
    use super::*;
    use duration_string::DurationString;
    use rand::thread_rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use std::ffi::CStr;

    #[no_mangle]
    pub unsafe extern "C" fn Java_com_example_zprize_RustMSM_benchmarkMSMRandom(
        env: JNIEnv,
        _: JClass,
        java_dir: JString,
        java_iters: JString,
        java_num_elems: JString,
    ) -> jstring {
        let mut rng = thread_rng();
        let base: i32 = 2;

        let num_elems = env
            .get_string(java_num_elems)
            .expect("invalid string")
            .as_ptr();
        let rust_num_elems = CStr::from_ptr(num_elems).to_str().expect("string invalid");
        let num_elems_val: u32 = rust_num_elems.parse().unwrap();
        let num_elems_exp = base.pow(num_elems_val);

        let instance = gen_random_vectors(num_elems_exp.try_into().unwrap(), &mut rng);
        let dir = env.get_string(java_dir).expect("invalid string").as_ptr();
        let rust_dir = CStr::from_ptr(dir).to_str().expect("string invalid");

        let iters = env.get_string(java_iters).expect("invalid string").as_ptr();
        let rust_iters = CStr::from_ptr(iters).to_str().expect("string invalid");
        let iters_val: u32 = rust_iters.parse().unwrap();

        let input_iter = VectorInputIterator::from(instance);
        let mean_time_vec = benchmark_msm(&rust_dir, input_iter, iters_val).unwrap();
        let mean_str: String = DurationString::from(mean_time_vec[0]).into();

        let output = env.new_string(&mean_str).unwrap();

        output.into_inner()
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_com_example_zprize_RustMSM_benchmarkMSMRandomMultipleVecs(
        env: JNIEnv,
        _: JClass,
        java_dir: JString,
        java_iters: JString,
        java_num_elems: JString,
        java_num_vecs: JString,
    ) -> jstring {
        let mut rng = thread_rng();
        let mut seed: [u8; 32] = [0; 32];
        rng.fill_bytes(&mut seed[..]);
        let mut chacha_rng = ChaCha20Rng::from_seed(seed);
        let base: i32 = 2;

        let num_elems = env
            .get_string(java_num_elems)
            .expect("invalid string")
            .as_ptr();
        let rust_num_elems = CStr::from_ptr(num_elems).to_str().expect("string invalid");
        let num_elems_val: u32 = rust_num_elems.parse().unwrap();
        let num_elems_exp = base.pow(num_elems_val);

        let num_vecs = env
            .get_string(java_num_vecs)
            .expect("invalid string")
            .as_ptr();
        let rust_num_vecs = CStr::from_ptr(num_vecs).to_str().expect("string invalid");
        let num_vecs_val: u32 = rust_num_vecs.parse().unwrap();

        let iters = env.get_string(java_iters).expect("invalid string").as_ptr();
        let rust_iters = CStr::from_ptr(iters).to_str().expect("string invalid");
        let iters_val: u32 = rust_iters.parse().unwrap();

        let mut points_vec = Vec::new();
        let mut scalars_vec = Vec::new();
        for _i in 0..num_vecs_val {
            let (points, scalars) =
                gen_random_vectors(num_elems_exp.try_into().unwrap(), &mut chacha_rng);
            points_vec.push(points);
            scalars_vec.push(scalars);
        }

        let dir = env.get_string(java_dir).expect("invalid string").as_ptr();
        let rust_dir = CStr::from_ptr(dir).to_str().expect("string invalid");
        let input_iter = VectorInputIterator::from((points_vec, scalars_vec));
        let mean_time_vec = benchmark_msm(&rust_dir, input_iter, iters_val).unwrap();

        let mut total = Duration::ZERO;
        for time in mean_time_vec {
            total += time;
        }
        let total_mean = total / num_vecs_val;

        let output_path = format!("{}{}", rust_dir, "/resulttimes.txt");
        let mut output_file = File::options().append(true).open(output_path).unwrap();
        writeln!(output_file, "Mean across all vectors: {:?}", total_mean).unwrap();
        let mean_str: String = DurationString::from(total_mean).into();
        let output = env.new_string(&mean_str).unwrap();

        output.into_inner()
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_com_example_zprize_RustMSM_benchmarkMSMFile(
        env: JNIEnv,
        _: JClass,
        java_dir: JString,
        java_iters: JString,
    ) -> jstring {
        let dir = env.get_string(java_dir).expect("invalid string").as_ptr();
        let rust_dir = CStr::from_ptr(dir).to_str().expect("string invalid");

        let iters = env.get_string(java_iters).expect("invalid string").as_ptr();
        let rust_iters = CStr::from_ptr(iters).to_str().expect("string invalid");
        let iters_val: u32 = rust_iters.parse().unwrap();

        let input_iter = FileInputIterator::open(&rust_dir).unwrap();
        let mean_time = benchmark_msm(&rust_dir, input_iter, iters_val).unwrap();

        let mut output_string = "".to_owned();
        for time in mean_time {
            let s: String = DurationString::from(time).into();
            output_string.push_str(&s);
            output_string.push_str(", ");
        }
        let mut output_chars = output_string.chars();
        output_chars.next_back();
        output_chars.next_back();

        let output = env.new_string(&output_chars.as_str()).unwrap();

        output.into_inner()
    }
}
