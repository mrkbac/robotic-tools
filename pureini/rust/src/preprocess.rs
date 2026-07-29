use std::borrow::Cow;
use std::cmp::Ordering;
use std::mem::{ManuallyDrop, MaybeUninit};

use rayon::prelude::*;

use crate::codec::{self, HeaderEncoding};
use crate::types::{EncodingInfo, FieldType, PointField};

const MAX_COUNTING_BUCKETS: usize = 4096;

pub struct EncodeResult {
    pub data: Vec<u8>,
    pub transformed_point_count: Option<u32>,
    pub did_filter_invalid_xyz: bool,
}

struct PreparedCloud<'a> {
    data: Cow<'a, [u8]>,
    transformed_point_count: Option<u32>,
    did_filter_invalid_xyz: bool,
}

pub fn encode(
    info: &EncodingInfo,
    cloud_data: &[u8],
    drop_invalid: bool,
    sort_field: Option<&str>,
) -> Result<EncodeResult, String> {
    let prepared = prepare(info, cloud_data, drop_invalid, sort_field)?;
    let mut transformed_info = None;
    let output_info = if let Some(point_count) = prepared.transformed_point_count {
        transformed_info.insert(EncodingInfo {
            width: point_count,
            height: 1,
            ..info.clone()
        })
    } else {
        info
    };
    let header = codec::encode_header(output_info, HeaderEncoding::Yaml)?;
    let data = codec::encode(output_info, &header, &prepared.data)?;
    Ok(EncodeResult {
        data,
        transformed_point_count: prepared.transformed_point_count,
        did_filter_invalid_xyz: prepared.did_filter_invalid_xyz,
    })
}

pub enum PreprocessOutput {
    Unchanged,
    Changed(Vec<u8>, u32),
}

pub struct PreprocessResult {
    pub output: PreprocessOutput,
    pub did_filter_invalid_xyz: bool,
}

pub fn preprocess(
    info: &EncodingInfo,
    cloud_data: &[u8],
    drop_invalid: bool,
    sort_field: Option<&str>,
) -> Result<PreprocessResult, String> {
    let prepared = prepare(info, cloud_data, drop_invalid, sort_field)?;
    let output = match prepared.data {
        Cow::Borrowed(_) => PreprocessOutput::Unchanged,
        Cow::Owned(data) => PreprocessOutput::Changed(
            data,
            prepared
                .transformed_point_count
                .expect("owned preprocessing output has a point count"),
        ),
    };
    Ok(PreprocessResult {
        output,
        did_filter_invalid_xyz: prepared.did_filter_invalid_xyz,
    })
}

fn prepare<'a>(
    info: &EncodingInfo,
    cloud_data: &'a [u8],
    drop_invalid: bool,
    sort_field: Option<&str>,
) -> Result<PreparedCloud<'a>, String> {
    let point_step = info.point_step as usize;
    if point_step == 0 {
        return Err("point_step must be greater than zero".to_string());
    }
    if !cloud_data.len().is_multiple_of(point_step) {
        return Err("Input cloud_data size is not a multiple of point_step".to_string());
    }

    let point_count = cloud_data.len() / point_step;
    let expected_point_count = (info.width as usize)
        .checked_mul(info.height as usize)
        .ok_or_else(|| "CloudINI point count overflows usize".to_string())?;
    if point_count != expected_point_count {
        return Err(format!(
            "Input cloud_data point count {point_count} does not match width * height \
             ({expected_point_count})"
        ));
    }
    let xyz = drop_invalid.then(|| xyz_fields(info)).flatten();
    let did_filter_invalid_xyz = xyz.is_some();
    let sort_field = sort_field
        .and_then(|name| info.fields.iter().find(|field| field.name == name))
        .filter(|field| field_fits(field, point_step));

    if point_count > 1
        && let Some(field) = sort_field
        && let Some((prepared, output_count)) =
            counting_group_points(cloud_data, point_step, field, xyz)
    {
        let output_count = u32::try_from(output_count)
            .map_err(|_| "Preprocessed point count exceeds u32".to_string())?;
        return Ok(PreparedCloud {
            data: Cow::Owned(prepared),
            transformed_point_count: Some(output_count),
            did_filter_invalid_xyz,
        });
    }

    if point_count > 1
        && sort_field.is_none()
        && let Some(xyz) = xyz
    {
        const PARALLEL_FILTER_THRESHOLD: usize = 256 * 1024;
        let (prepared, output_count) = if point_count >= PARALLEL_FILTER_THRESHOLD {
            filter_points_parallel(cloud_data, point_step, xyz)
        } else {
            filter_points_sequential(cloud_data, point_step, xyz)
        };
        let Some(prepared) = prepared else {
            return Ok(PreparedCloud {
                data: Cow::Borrowed(cloud_data),
                transformed_point_count: None,
                did_filter_invalid_xyz,
            });
        };
        let output_count = u32::try_from(output_count)
            .map_err(|_| "Preprocessed point count exceeds u32".to_string())?;
        return Ok(PreparedCloud {
            data: Cow::Owned(prepared),
            transformed_point_count: Some(output_count),
            did_filter_invalid_xyz,
        });
    }

    let mut indices = (0..point_count).collect::<Vec<_>>();
    let mut was_transformed = false;

    if let Some(xyz) = xyz {
        let original_count = indices.len();
        indices.retain(|&index| {
            let point = &cloud_data[index * point_step..(index + 1) * point_step];
            !xyz.is_invalid(point)
        });
        was_transformed = indices.len() != original_count;
    }

    if let Some(field) = sort_field
        && indices.len() > 1
    {
        indices.sort_by(|&left, &right| compare_field(cloud_data, point_step, field, left, right));
        was_transformed = true;
    }

    if was_transformed {
        let output_count = u32::try_from(indices.len())
            .map_err(|_| "Preprocessed point count exceeds u32".to_string())?;
        Ok(PreparedCloud {
            data: Cow::Owned(gather_points(cloud_data, point_step, &indices)),
            transformed_point_count: Some(output_count),
            did_filter_invalid_xyz,
        })
    } else {
        Ok(PreparedCloud {
            data: Cow::Borrowed(cloud_data),
            transformed_point_count: None,
            did_filter_invalid_xyz,
        })
    }
}

fn filter_points_sequential(
    cloud_data: &[u8],
    point_step: usize,
    xyz: XyzFields,
) -> (Option<Vec<u8>>, usize) {
    let mut output = None;
    let mut output_length = 0;
    for (index, point) in cloud_data.chunks_exact(point_step).enumerate() {
        if xyz.is_invalid(point) {
            if output.is_none() {
                let mut initialized = vec![MaybeUninit::<u8>::uninit(); cloud_data.len()];
                output_length = index * point_step;
                // Every point before the first invalid point is retained as one contiguous prefix.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        cloud_data.as_ptr(),
                        initialized.as_mut_ptr().cast::<u8>(),
                        output_length,
                    );
                }
                output = Some(initialized);
            }
            continue;
        }
        let Some(output) = &mut output else {
            continue;
        };
        // `output_length` advances by one point only after this disjoint slot is written.
        unsafe {
            std::ptr::copy_nonoverlapping(
                point.as_ptr(),
                output.as_mut_ptr().add(output_length).cast::<u8>(),
                point_step,
            );
        }
        output_length += point_step;
    }
    let Some(mut output) = output else {
        return (None, cloud_data.len() / point_step);
    };
    output.truncate(output_length);
    (Some(assume_init_bytes(output)), output_length / point_step)
}

struct FilterChunk {
    data_offset: usize,
    point_count: usize,
    output_point: usize,
}

fn filter_points_parallel(
    cloud_data: &[u8],
    point_step: usize,
    xyz: XyzFields,
) -> (Option<Vec<u8>>, usize) {
    const CHUNK_POINTS: usize = 64 * 1024;

    let chunk_bytes = CHUNK_POINTS * point_step;
    let mut chunks = cloud_data
        .par_chunks(chunk_bytes)
        .enumerate()
        .map(|(chunk_index, data)| {
            let retained = data
                .chunks_exact(point_step)
                .filter(|point| !xyz.is_invalid(point))
                .count();
            (
                FilterChunk {
                    data_offset: chunk_index * chunk_bytes,
                    point_count: data.len() / point_step,
                    output_point: 0,
                },
                retained,
            )
        })
        .collect::<Vec<_>>();

    let mut output_count = 0;
    for (chunk, retained) in &mut chunks {
        chunk.output_point = output_count;
        output_count += *retained;
    }
    if output_count == cloud_data.len() / point_step {
        return (None, output_count);
    }
    let output_length = output_count * point_step;
    let mut output = vec![MaybeUninit::<u8>::uninit(); output_length];
    let output_pointer = output.as_mut_ptr() as usize;
    chunks.into_par_iter().for_each(|(mut chunk, _)| {
        let data_end = chunk.data_offset + chunk.point_count * point_step;
        let data = &cloud_data[chunk.data_offset..data_end];
        for point in data.chunks_exact(point_step) {
            if xyz.is_invalid(point) {
                continue;
            }
            let offset = chunk.output_point * point_step;
            // Chunk prefix ranges are disjoint and each retained slot is written once.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    point.as_ptr(),
                    (output_pointer as *mut u8).add(offset),
                    point_step,
                );
            }
            chunk.output_point += 1;
        }
    });
    (Some(assume_init_bytes(output)), output_count)
}

#[derive(Clone, Copy)]
struct NumericField {
    offset: usize,
    field_type: FieldType,
}

impl NumericField {
    #[inline(always)]
    fn read(self, point: &[u8]) -> f64 {
        let data = &point[self.offset..];
        match self.field_type {
            FieldType::Int8 => f64::from(data[0] as i8),
            FieldType::Uint8 => f64::from(data[0]),
            FieldType::Int16 => f64::from(read_i16(data)),
            FieldType::Uint16 => f64::from(read_u16(data)),
            FieldType::Int32 => f64::from(read_i32(data)),
            FieldType::Uint32 => f64::from(read_u32(data)),
            FieldType::Float32 => f64::from(read_f32(data, 0)),
            FieldType::Float64 => read_f64(data),
            FieldType::Int64 => read_i64(data) as f64,
            FieldType::Uint64 => read_u64(data) as f64,
            FieldType::Unknown => unreachable!("unknown fields are rejected by numeric_field"),
        }
    }
}

#[derive(Clone, Copy)]
enum XyzFields {
    Float32([usize; 3]),
    Float64([usize; 3]),
    Mixed([NumericField; 3]),
}

impl XyzFields {
    #[inline(always)]
    fn is_invalid(self, point: &[u8]) -> bool {
        // `xyz_fields` rejects offsets that do not fit within a point.
        match self {
            Self::Float32([x, y, z]) => {
                let x = unsafe { read_u32_at(point, x) };
                let y = unsafe { read_u32_at(point, y) };
                let z = unsafe { read_u32_at(point, z) };
                is_non_finite_f32_bits(x)
                    || is_non_finite_f32_bits(y)
                    || is_non_finite_f32_bits(z)
                    || (is_zero_f32_bits(x) && is_zero_f32_bits(y) && is_zero_f32_bits(z))
            }
            Self::Float64([x, y, z]) => {
                let x = unsafe { read_u64_at(point, x) };
                let y = unsafe { read_u64_at(point, y) };
                let z = unsafe { read_u64_at(point, z) };
                is_non_finite_f64_bits(x)
                    || is_non_finite_f64_bits(y)
                    || is_non_finite_f64_bits(z)
                    || (is_zero_f64_bits(x) && is_zero_f64_bits(y) && is_zero_f64_bits(z))
            }
            Self::Mixed([x, y, z]) => {
                let x = x.read(point);
                let y = y.read(point);
                let z = z.read(point);
                !x.is_finite()
                    || !y.is_finite()
                    || !z.is_finite()
                    || (x == 0.0 && y == 0.0 && z == 0.0)
            }
        }
    }
}

#[inline(always)]
unsafe fn read_u32_at(data: &[u8], offset: usize) -> u32 {
    u32::from_le(unsafe { data.as_ptr().add(offset).cast::<u32>().read_unaligned() })
}

#[inline(always)]
unsafe fn read_u64_at(data: &[u8], offset: usize) -> u64 {
    u64::from_le(unsafe { data.as_ptr().add(offset).cast::<u64>().read_unaligned() })
}

#[inline(always)]
fn is_non_finite_f32_bits(value: u32) -> bool {
    value & 0x7f80_0000 == 0x7f80_0000
}

#[inline(always)]
fn is_zero_f32_bits(value: u32) -> bool {
    value & 0x7fff_ffff == 0
}

#[inline(always)]
fn is_non_finite_f64_bits(value: u64) -> bool {
    value & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000
}

#[inline(always)]
fn is_zero_f64_bits(value: u64) -> bool {
    value & 0x7fff_ffff_ffff_ffff == 0
}

fn xyz_fields(info: &EncodingInfo) -> Option<XyzFields> {
    let point_step = info.point_step as usize;
    let field = |name| {
        info.fields
            .iter()
            .find(|field| field.name == name)
            .and_then(|field| numeric_field(field, point_step))
    };
    let fields = [field("x")?, field("y")?, field("z")?];
    let offsets = [fields[0].offset, fields[1].offset, fields[2].offset];
    if fields
        .iter()
        .all(|field| field.field_type == FieldType::Float32)
    {
        Some(XyzFields::Float32(offsets))
    } else if fields
        .iter()
        .all(|field| field.field_type == FieldType::Float64)
    {
        Some(XyzFields::Float64(offsets))
    } else {
        Some(XyzFields::Mixed(fields))
    }
}

fn numeric_field(field: &PointField, point_step: usize) -> Option<NumericField> {
    (field.field_type != FieldType::Unknown && field_fits(field, point_step)).then_some(
        NumericField {
            offset: field.offset as usize,
            field_type: field.field_type,
        },
    )
}

fn field_fits(field: &PointField, point_step: usize) -> bool {
    let offset = field.offset as usize;
    let size = field.field_type.size_of();
    size != 0
        && offset
            .checked_add(size)
            .is_some_and(|end| end <= point_step)
}

fn gather_points(cloud_data: &[u8], point_step: usize, indices: &[usize]) -> Vec<u8> {
    let mut output = Vec::with_capacity(indices.len() * point_step);
    for &index in indices {
        let offset = index * point_step;
        output.extend_from_slice(&cloud_data[offset..offset + point_step]);
    }
    output
}

fn counting_group_points(
    cloud_data: &[u8],
    point_step: usize,
    field: &PointField,
    xyz: Option<XyzFields>,
) -> Option<(Vec<u8>, usize)> {
    const PARALLEL_THRESHOLD: usize = 64 * 1024;

    if cloud_data.len() / point_step >= PARALLEL_THRESHOLD {
        return counting_group_points_parallel(cloud_data, point_step, field, xyz);
    }
    counting_group_points_sequential(cloud_data, point_step, field, xyz)
}

fn counting_group_points_sequential(
    cloud_data: &[u8],
    point_step: usize,
    field: &PointField,
    xyz: Option<XyzFields>,
) -> Option<(Vec<u8>, usize)> {
    const DROPPED: u16 = u16::MAX;

    let bucket_count = counting_bucket_count(field);
    let mut counts = vec![0_usize; bucket_count];
    let mut buckets = vec![MaybeUninit::<u16>::uninit(); cloud_data.len() / point_step];
    let mut output_count = 0_usize;
    for (index, point) in cloud_data.chunks_exact(point_step).enumerate() {
        if xyz.is_some_and(|xyz| xyz.is_invalid(point)) {
            buckets[index].write(DROPPED);
            continue;
        }
        let key = nonnegative_integer_key(point, field)?;
        if key >= bucket_count {
            return None;
        }
        buckets[index].write(key as u16);
        counts[key] += 1;
        output_count += 1;
    }
    let mut positions = vec![0_usize; bucket_count];
    let mut total = 0_usize;
    for (position, count) in positions.iter_mut().zip(counts) {
        *position = total;
        total += count;
    }

    let output_length = output_count * point_step;
    let mut output = vec![MaybeUninit::<u8>::uninit(); output_length];
    for (point, bucket) in cloud_data.chunks_exact(point_step).zip(buckets) {
        // The first pass writes one bucket value for every input point.
        let bucket = unsafe { bucket.assume_init() };
        if bucket == DROPPED {
            continue;
        }
        let key = bucket as usize;
        let offset = positions[key] * point_step;
        // Prefix-sum positions are disjoint and cover every retained output slot once.
        unsafe {
            std::ptr::copy_nonoverlapping(
                point.as_ptr(),
                output.as_mut_ptr().add(offset).cast::<u8>(),
                point_step,
            );
        }
        positions[key] += 1;
    }
    Some((assume_init_bytes(output), output_count))
}

struct BucketChunk {
    data_offset: usize,
    point_count: usize,
    buckets: Vec<u16>,
    counts: Vec<usize>,
    positions: Vec<usize>,
}

fn counting_group_points_parallel(
    cloud_data: &[u8],
    point_step: usize,
    field: &PointField,
    xyz: Option<XyzFields>,
) -> Option<(Vec<u8>, usize)> {
    const DROPPED: u16 = u16::MAX;
    const CHUNK_POINTS: usize = 64 * 1024;

    let bucket_count = counting_bucket_count(field);
    let chunk_bytes = CHUNK_POINTS * point_step;
    let mut chunks = cloud_data
        .par_chunks(chunk_bytes)
        .enumerate()
        .map(|(chunk_index, data)| {
            let point_count = data.len() / point_step;
            let mut buckets = vec![DROPPED; point_count];
            let mut counts = vec![0_usize; bucket_count];
            for (index, point) in data.chunks_exact(point_step).enumerate() {
                if xyz.is_some_and(|xyz| xyz.is_invalid(point)) {
                    continue;
                }
                let key = nonnegative_integer_key(point, field)?;
                if key >= bucket_count {
                    return None;
                }
                buckets[index] = key as u16;
                counts[key] += 1;
            }
            Some(BucketChunk {
                data_offset: chunk_index * chunk_bytes,
                point_count,
                buckets,
                counts,
                positions: vec![0_usize; bucket_count],
            })
        })
        .collect::<Option<Vec<_>>>()?;

    let mut totals = vec![0_usize; bucket_count];
    for chunk in &chunks {
        for (total, count) in totals.iter_mut().zip(chunk.counts.iter()) {
            *total += *count;
        }
    }
    let output_count = totals.iter().sum::<usize>();
    let mut next_position = vec![0_usize; bucket_count];
    let mut total = 0_usize;
    for (position, count) in next_position.iter_mut().zip(totals) {
        *position = total;
        total += count;
    }
    for chunk in &mut chunks {
        chunk.positions.copy_from_slice(&next_position);
        for (position, count) in next_position.iter_mut().zip(chunk.counts.iter()) {
            *position += *count;
        }
    }

    let output_length = output_count * point_step;
    let mut output = vec![MaybeUninit::<u8>::uninit(); output_length];
    let output_pointer = output.as_mut_ptr() as usize;
    chunks.into_par_iter().for_each(|mut chunk| {
        let data_end = chunk.data_offset + chunk.point_count * point_step;
        let data = &cloud_data[chunk.data_offset..data_end];
        for (point, bucket) in data.chunks_exact(point_step).zip(chunk.buckets) {
            if bucket == DROPPED {
                continue;
            }
            let key = bucket as usize;
            let offset = chunk.positions[key] * point_step;
            // Per-chunk prefix ranges are disjoint and cover each retained slot once.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    point.as_ptr(),
                    (output_pointer as *mut u8).add(offset),
                    point_step,
                );
            }
            chunk.positions[key] += 1;
        }
    });
    Some((assume_init_bytes(output), output_count))
}

fn counting_bucket_count(field: &PointField) -> usize {
    match field.field_type {
        FieldType::Int8 => 128,
        FieldType::Uint8 => 256,
        _ => MAX_COUNTING_BUCKETS,
    }
}

fn assume_init_bytes(output: Vec<MaybeUninit<u8>>) -> Vec<u8> {
    let mut output = ManuallyDrop::new(output);
    // Every element was initialized by `counting_group_points`, and u8 has identical layout.
    unsafe {
        Vec::from_raw_parts(
            output.as_mut_ptr().cast::<u8>(),
            output.len(),
            output.capacity(),
        )
    }
}

#[inline(always)]
fn nonnegative_integer_key(point: &[u8], field: &PointField) -> Option<usize> {
    let data = &point[field.offset as usize..];
    match field.field_type {
        FieldType::Int8 => usize::try_from(data[0] as i8).ok(),
        FieldType::Uint8 => Some(data[0] as usize),
        FieldType::Int16 => usize::try_from(read_i16(data)).ok(),
        FieldType::Uint16 => Some(read_u16(data) as usize),
        FieldType::Int32 => usize::try_from(read_i32(data)).ok(),
        FieldType::Uint32 => usize::try_from(read_u32(data)).ok(),
        FieldType::Int64 => usize::try_from(read_i64(data)).ok(),
        FieldType::Uint64 => usize::try_from(read_u64(data)).ok(),
        FieldType::Float32 => integral_float_key(f64::from(read_f32(data, 0))),
        FieldType::Float64 => integral_float_key(read_f64(data)),
        FieldType::Unknown => None,
    }
}

#[inline(always)]
fn integral_float_key(value: f64) -> Option<usize> {
    (value.is_finite()
        && value >= 0.0
        && value < MAX_COUNTING_BUCKETS as f64
        && value.trunc() == value)
        .then_some(value as usize)
}

fn compare_field(
    cloud_data: &[u8],
    point_step: usize,
    field: &PointField,
    left: usize,
    right: usize,
) -> Ordering {
    let offset = field.offset as usize;
    let left = &cloud_data[left * point_step + offset..];
    let right = &cloud_data[right * point_step + offset..];
    match field.field_type {
        FieldType::Int8 => (left[0] as i8).cmp(&(right[0] as i8)),
        FieldType::Uint8 => left[0].cmp(&right[0]),
        FieldType::Int16 => read_i16(left).cmp(&read_i16(right)),
        FieldType::Uint16 => read_u16(left).cmp(&read_u16(right)),
        FieldType::Int32 => read_i32(left).cmp(&read_i32(right)),
        FieldType::Uint32 => read_u32(left).cmp(&read_u32(right)),
        FieldType::Int64 => read_i64(left).cmp(&read_i64(right)),
        FieldType::Uint64 => read_u64(left).cmp(&read_u64(right)),
        FieldType::Float32 => compare_float(read_f32(left, 0) as f64, read_f32(right, 0) as f64),
        FieldType::Float64 => compare_float(read_f64(left), read_f64(right)),
        FieldType::Unknown => Ordering::Equal,
    }
}

fn compare_float(left: f64, right: f64) -> Ordering {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
    }
}

#[inline(always)]
fn read_f32(data: &[u8], offset: usize) -> f32 {
    // Callers validate the field range against point_step.
    let bits = unsafe { data.as_ptr().add(offset).cast::<u32>().read_unaligned() };
    f32::from_bits(u32::from_le(bits))
}

#[inline(always)]
fn read_f64(data: &[u8]) -> f64 {
    let bits = unsafe { data.as_ptr().cast::<u64>().read_unaligned() };
    f64::from_bits(u64::from_le(bits))
}

#[inline(always)]
fn read_i16(data: &[u8]) -> i16 {
    i16::from_le(unsafe { data.as_ptr().cast::<i16>().read_unaligned() })
}

#[inline(always)]
fn read_u16(data: &[u8]) -> u16 {
    u16::from_le(unsafe { data.as_ptr().cast::<u16>().read_unaligned() })
}

#[inline(always)]
fn read_i32(data: &[u8]) -> i32 {
    i32::from_le(unsafe { data.as_ptr().cast::<i32>().read_unaligned() })
}

#[inline(always)]
fn read_u32(data: &[u8]) -> u32 {
    u32::from_le(unsafe { data.as_ptr().cast::<u32>().read_unaligned() })
}

#[inline(always)]
fn read_i64(data: &[u8]) -> i64 {
    i64::from_le(unsafe { data.as_ptr().cast::<i64>().read_unaligned() })
}

#[inline(always)]
fn read_u64(data: &[u8]) -> u64 {
    u64::from_le(unsafe { data.as_ptr().cast::<u64>().read_unaligned() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonnegative_integer_key_accepts_integral_float_buckets() {
        let float32 = PointField {
            name: "ring".to_string(),
            offset: 0,
            field_type: FieldType::Float32,
            resolution: None,
        };
        let float64 = PointField {
            field_type: FieldType::Float64,
            ..float32.clone()
        };

        assert_eq!(
            nonnegative_integer_key(&127.0_f32.to_le_bytes(), &float32),
            Some(127)
        );
        assert_eq!(
            nonnegative_integer_key(&4095.0_f64.to_le_bytes(), &float64),
            Some(4095)
        );
    }

    #[test]
    fn nonnegative_integer_key_rejects_non_bucket_floats() {
        let field = PointField {
            name: "ring".to_string(),
            offset: 0,
            field_type: FieldType::Float32,
            resolution: None,
        };

        for value in [-1.0_f32, 1.5, 4096.0, f32::NAN, f32::INFINITY] {
            assert_eq!(nonnegative_integer_key(&value.to_le_bytes(), &field), None);
        }
    }
}
