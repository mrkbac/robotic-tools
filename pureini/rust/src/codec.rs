use std::collections::HashMap;

use crate::field_codec::{
    FieldDecoder, FieldEncoder, build_decoders, build_encoders, decode_multi_points,
    encode_contiguous_xyz_xor, encode_contiguous_xyz_xor_into, encode_dense_u32_points,
    encode_multi_points, read_integer_kind,
};
use crate::types::{
    CompressionOption, EncodingInfo, EncodingOptions, FieldType, POINTS_PER_CHUNK, PointField,
};
use crate::varint::{
    append as append_varint, decode_at as decode_varint_at, encode as encode_varint64,
};

pub const ENCODING_VERSION: u8 = 5;
const MAGIC_HEADER: &[u8; 10] = b"CLOUDINI_V";
const ADAPTIVE_MODE_PROBE_POINTS: usize = 4096;

pub type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderEncoding {
    Binary,
    Yaml,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum AdaptiveMode {
    DeltaVarint = 0,
    Palette = 1,
    Rle = 2,
    DeltaRle = 3,
}

#[derive(Clone)]
struct AdaptiveField {
    offset: usize,
    field_type: FieldType,
    size: usize,
    committed_mode: Option<AdaptiveMode>,
}

pub fn encoding_info_to_yaml(info: &EncodingInfo) -> String {
    let mut yaml = format!(
        "version: {}\nwidth: {}\nheight: {}\npoint_step: {}\nencoding_opt: \
         {}\ncompression_opt: {}\n",
        info.version,
        info.width,
        info.height,
        info.point_step,
        info.encoding_opt.as_str(),
        info.compression_opt.as_str(),
    );
    if !info.encoding_config.is_empty() {
        yaml.push_str(&format!("encoding_config: {}\n", info.encoding_config));
    }
    yaml.push_str("fields:\n");
    for field in &info.fields {
        yaml.push_str(&format!(
            "  - name: {}\n    offset: {}\n    type: {}\n    resolution: {}\n",
            field.name,
            field.offset,
            field.field_type.as_str(),
            field
                .resolution
                .map_or_else(|| "null".to_string(), |value| value.to_string()),
        ));
    }
    yaml
}

pub fn encoding_info_from_yaml(yaml: &str) -> Result<EncodingInfo> {
    let mut info = EncodingInfo::default();
    let mut current_field: Option<PointField> = None;
    for line in yaml.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(field_line) = trimmed.strip_prefix("- ") {
            if let Some(field) = current_field.take() {
                info.fields.push(field);
            }
            current_field = Some(PointField {
                name: String::new(),
                offset: 0,
                field_type: FieldType::Unknown,
                resolution: None,
            });
            apply_field_yaml(
                current_field.as_mut().unwrap(),
                split_yaml_entry(field_line)?,
            )?;
            continue;
        }

        let (key, value) = split_yaml_entry(trimmed)?;
        if let Some(field) = current_field.as_mut()
            && matches!(key, "name" | "offset" | "type" | "resolution")
        {
            apply_field_yaml(field, (key, value))?;
            continue;
        }
        match key {
            "version" => {
                info.version = parse_yaml_number(value, key)?;
            }
            "width" => info.width = parse_yaml_number(value, key)?,
            "height" => info.height = parse_yaml_number(value, key)?,
            "point_step" => info.point_step = parse_yaml_number(value, key)?,
            "encoding_opt" => {
                info.encoding_opt = EncodingOptions::from_str(value)
                    .ok_or_else(|| format!("Invalid encoding_opt '{value}'"))?;
            }
            "compression_opt" => {
                info.compression_opt = CompressionOption::from_str(value)
                    .ok_or_else(|| format!("Invalid compression_opt '{value}'"))?;
            }
            "encoding_config" => info.encoding_config = value.to_string(),
            "fields" => {}
            _ => {}
        }
    }
    if let Some(field) = current_field {
        info.fields.push(field);
    }
    Ok(info)
}

pub fn encode_header(info: &EncodingInfo, encoding: HeaderEncoding) -> Result<Vec<u8>> {
    if info.version > 99 {
        return Err(format!(
            "CloudINI versions must fit in two decimal digits, got {}",
            info.version
        ));
    }

    match encoding {
        HeaderEncoding::Yaml => {
            let yaml = encoding_info_to_yaml(info);
            let mut output = Vec::with_capacity(MAGIC_HEADER.len() + 3 + yaml.len() + 1);
            write_magic(info.version, &mut output);
            output.push(b'\n');
            output.extend_from_slice(yaml.as_bytes());
            output.push(0);
            Ok(output)
        }
        HeaderEncoding::Binary => {
            let mut output = Vec::new();
            write_magic(info.version, &mut output);
            output.extend_from_slice(&info.width.to_le_bytes());
            output.extend_from_slice(&info.height.to_le_bytes());
            output.extend_from_slice(&info.point_step.to_le_bytes());
            output.push(info.encoding_opt as u8);
            output.push(info.compression_opt as u8);
            let field_count = u16::try_from(info.fields.len())
                .map_err(|_| "CloudINI headers support at most 65535 fields".to_string())?;
            output.extend_from_slice(&field_count.to_le_bytes());
            for field in &info.fields {
                let name = field.name.as_bytes();
                let name_len = u16::try_from(name.len())
                    .map_err(|_| "CloudINI field names must fit in 65535 bytes".to_string())?;
                output.extend_from_slice(&name_len.to_le_bytes());
                output.extend_from_slice(name);
                output.extend_from_slice(&field.offset.to_le_bytes());
                output.push(field.field_type as u8);
                output.extend_from_slice(&field.resolution.unwrap_or(-1.0).to_le_bytes());
            }
            Ok(output)
        }
    }
}

pub fn decode_header(data: &[u8]) -> Result<(EncodingInfo, usize)> {
    if data.len() < MAGIC_HEADER.len() + 2 {
        return Err("Input too small to contain CloudINI header".to_string());
    }
    if &data[..MAGIC_HEADER.len()] != MAGIC_HEADER {
        return Err(format!(
            "Invalid magic header. Expected 'CLOUDINI_V', got: '{}'",
            String::from_utf8_lossy(&data[..MAGIC_HEADER.len()])
        ));
    }

    let version_digits = &data[MAGIC_HEADER.len()..MAGIC_HEADER.len() + 2];
    if !version_digits.iter().all(u8::is_ascii_digit) {
        return Err("CloudINI version is not two ASCII digits".to_string());
    }
    let version = (version_digits[0] - b'0') * 10 + version_digits[1] - b'0';
    if !(2..=ENCODING_VERSION).contains(&version) {
        return Err(format!(
            "Unsupported encoding version. Current is: {ENCODING_VERSION}, got: {version}"
        ));
    }

    let mut position = MAGIC_HEADER.len() + 2;
    if data.get(position) == Some(&b'\n') && data.get(position + 1) != Some(&b'{') {
        position += 1;
        let terminator = data[position..]
            .iter()
            .position(|byte| *byte == 0)
            .ok_or_else(|| "Malformed YAML header: missing null terminator".to_string())?;
        let yaml = std::str::from_utf8(&data[position..position + terminator])
            .map_err(|error| format!("Malformed YAML header: {error}"))?;
        let mut info = encoding_info_from_yaml(yaml)?;
        info.version = version;
        return Ok((info, position + terminator + 1));
    }

    let mut info = EncodingInfo {
        version,
        ..EncodingInfo::default()
    };
    info.width = read_u32(data, &mut position, "width")?;
    info.height = read_u32(data, &mut position, "height")?;
    info.point_step = read_u32(data, &mut position, "point_step")?;
    info.encoding_opt = encoding_option(read_u8(data, &mut position, "encoding option")?)?;
    info.compression_opt = compression_option(read_u8(data, &mut position, "compression option")?)?;
    let field_count = read_u16(data, &mut position, "field count")?;
    for _ in 0..field_count {
        let name_len = usize::from(read_u16(data, &mut position, "field name length")?);
        let name_bytes = read_slice(data, &mut position, name_len, "field name")?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|error| format!("Invalid UTF-8 field name: {error}"))?
            .to_string();
        let offset = read_u32(data, &mut position, "field offset")?;
        let field_type = field_type(read_u8(data, &mut position, "field type")?)?;
        let resolution =
            f32::from_le_bytes(read_array::<4>(data, &mut position, "field resolution")?);
        info.fields.push(PointField {
            name,
            offset,
            field_type,
            resolution: (resolution > 0.0).then_some(resolution),
        });
    }
    Ok((info, position))
}

pub fn encode(info: &EncodingInfo, header: &[u8], cloud_data: &[u8]) -> Result<Vec<u8>> {
    validate_info(info)?;
    if !cloud_data.len().is_multiple_of(info.point_step as usize) {
        return Err("Input cloud_data size is not a multiple of point_step".to_string());
    }
    let point_step = info.point_step as usize;
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
    if uses_direct_lossless_xyz(info) {
        return encode_direct_lossless_xyz(info, header, cloud_data);
    }

    let mut output = header.to_vec();
    validate_integer_varint_deltas(info, cloud_data, point_step)?;
    let mut compressor = ChunkCompressor::new(info.compression_opt)?;

    if !uses_v5_codec(info) {
        let mut encoders = build_encoders(&info.fields, info.encoding_opt, info.version);
        let chunk_limit = if info.version >= 3 {
            POINTS_PER_CHUNK
        } else {
            point_count.max(1)
        };
        let mut stage = Vec::with_capacity(point_count.min(chunk_limit) * point_step);
        for chunk_start in (0..point_count).step_by(chunk_limit) {
            let chunk_end = (chunk_start + chunk_limit).min(point_count);
            for encoder in &mut encoders {
                encoder.reset();
            }
            stage.clear();
            append_regular_chunk(
                &mut encoders,
                &cloud_data[chunk_start * point_step..chunk_end * point_step],
                point_step,
                &mut stage,
            );
            let chunk = compressor.compress(&stage)?;
            if chunk_start == 0 {
                let chunk_count = point_count.div_ceil(chunk_limit);
                let framing_size = usize::from(info.version >= 3) * 4;
                output.reserve(
                    chunk
                        .len()
                        .saturating_add(framing_size)
                        .saturating_mul(chunk_count),
                );
            }
            if info.version >= 3 {
                let chunk_len = u32::try_from(chunk.len())
                    .map_err(|_| "Compressed CloudINI chunk exceeds 4 GiB".to_string())?;
                output.extend_from_slice(&chunk_len.to_le_bytes());
            }
            output.extend_from_slice(chunk);
        }
        return Ok(output);
    }

    let regular_fields = regular_fields(info);
    let mut regular_encoders = build_encoders(&regular_fields, info.encoding_opt, info.version);
    let mut adaptive_fields = adaptive_fields(info);

    for chunk_start in (0..point_count).step_by(POINTS_PER_CHUNK) {
        let chunk_end = (chunk_start + POINTS_PER_CHUNK).min(point_count);
        let chunk_points = chunk_end - chunk_start;
        for encoder in &mut regular_encoders {
            encoder.reset();
        }

        let mut stage = Vec::with_capacity(chunk_points * point_step);
        let chunk_data = &cloud_data[chunk_start * point_step..chunk_end * point_step];
        if adaptive_fields
            .iter()
            .any(|field| field.committed_mode.is_none())
        {
            let probe_count = chunk_points.min(ADAPTIVE_MODE_PROBE_POINTS);
            for field in &mut adaptive_fields {
                if field.committed_mode.is_none() {
                    let values = chunk_data
                        .chunks_exact(point_step)
                        .take(probe_count)
                        .map(|point| {
                            read_integer(
                                &point[field.offset..field.offset + field.size],
                                field.field_type,
                            )
                        })
                        .collect::<Vec<_>>();
                    field.committed_mode = Some(select_mode(&values, field.size));
                }
            }
        }
        if !regular_encoders.is_empty() {
            append_regular_chunk(&mut regular_encoders, chunk_data, point_step, &mut stage);
        }
        for field in &adaptive_fields {
            append_streaming_adaptive_section(field, chunk_data, point_step, &mut stage)?;
        }

        let chunk = compressor.compress(&stage)?;
        let chunk_len = u32::try_from(chunk.len())
            .map_err(|_| "Compressed CloudINI chunk exceeds 4 GiB".to_string())?;
        output.extend_from_slice(&chunk_len.to_le_bytes());
        output.extend_from_slice(chunk);
    }

    Ok(output)
}

fn uses_direct_lossless_xyz(info: &EncodingInfo) -> bool {
    info.compression_opt == CompressionOption::None
        && info.encoding_opt == EncodingOptions::Lossless
        && info.point_step == 12
        && matches!(
            info.fields.as_slice(),
            [
                PointField {
                    offset: 0,
                    field_type: FieldType::Float32,
                    ..
                },
                PointField {
                    offset: 4,
                    field_type: FieldType::Float32,
                    ..
                },
                PointField {
                    offset: 8,
                    field_type: FieldType::Float32,
                    ..
                },
            ]
        )
}

fn encode_direct_lossless_xyz(
    info: &EncodingInfo,
    header: &[u8],
    cloud_data: &[u8],
) -> Result<Vec<u8>> {
    let point_count = cloud_data.len() / 12;
    let chunk_limit = if info.version >= 3 {
        POINTS_PER_CHUNK
    } else {
        point_count.max(1)
    };
    let output_length = direct_lossless_xyz_output_length(info, header.len(), cloud_data.len())
        .ok_or_else(|| "Encoded CloudINI size overflows usize".to_string())?;
    let mut output = Vec::with_capacity(output_length);
    output.extend_from_slice(header);
    for chunk_start in (0..point_count).step_by(chunk_limit) {
        let chunk_end = (chunk_start + chunk_limit).min(point_count);
        let chunk = &cloud_data[chunk_start * 12..chunk_end * 12];
        if info.version >= 3 {
            let chunk_len = u32::try_from(chunk.len())
                .map_err(|_| "Compressed CloudINI chunk exceeds 4 GiB".to_string())?;
            output.extend_from_slice(&chunk_len.to_le_bytes());
        }
        encode_contiguous_xyz_xor(chunk, &mut output, [0; 3]);
    }
    Ok(output)
}

pub(crate) fn direct_lossless_xyz_output_length(
    info: &EncodingInfo,
    header_length: usize,
    cloud_data_length: usize,
) -> Option<usize> {
    if !uses_direct_lossless_xyz(info) || !cloud_data_length.is_multiple_of(12) {
        return None;
    }
    let point_count = cloud_data_length / 12;
    if point_count != (info.width as usize).checked_mul(info.height as usize)? {
        return None;
    }
    let chunk_limit = if info.version >= 3 {
        POINTS_PER_CHUNK
    } else {
        point_count.max(1)
    };
    let framing_size = usize::from(info.version >= 3) * 4;
    let framing_length = point_count
        .div_ceil(chunk_limit)
        .checked_mul(framing_size)?;
    header_length
        .checked_add(cloud_data_length)?
        .checked_add(framing_length)
}

pub(crate) fn encode_direct_lossless_xyz_into(
    info: &EncodingInfo,
    header: &[u8],
    cloud_data: &[u8],
    output: &mut [u8],
) -> Result<()> {
    validate_info(info)?;
    let expected_length =
        direct_lossless_xyz_output_length(info, header.len(), cloud_data.len())
            .ok_or_else(|| "Direct lossless XYZ encoding is not applicable".to_string())?;
    if output.len() != expected_length {
        return Err("Direct lossless XYZ output buffer has the wrong size".to_string());
    }

    output[..header.len()].copy_from_slice(header);
    let point_count = cloud_data.len() / 12;
    let chunk_limit = if info.version >= 3 {
        POINTS_PER_CHUNK
    } else {
        point_count.max(1)
    };
    let mut output_position = header.len();
    for chunk_start in (0..point_count).step_by(chunk_limit) {
        let chunk_end = (chunk_start + chunk_limit).min(point_count);
        let chunk = &cloud_data[chunk_start * 12..chunk_end * 12];
        if info.version >= 3 {
            let chunk_len = u32::try_from(chunk.len())
                .map_err(|_| "Compressed CloudINI chunk exceeds 4 GiB".to_string())?;
            output[output_position..output_position + 4].copy_from_slice(&chunk_len.to_le_bytes());
            output_position += 4;
        }
        let chunk_output = &mut output[output_position..output_position + chunk.len()];
        encode_contiguous_xyz_xor_into(chunk, chunk_output, [0; 3]);
        output_position += chunk.len();
    }
    debug_assert_eq!(output_position, output.len());
    Ok(())
}

pub(crate) fn decode_payload_into(
    info: &EncodingInfo,
    payload: &[u8],
    output: &mut [u8],
) -> Result<()> {
    validate_info(info)?;
    let total_points = info.width as usize * info.height as usize;
    let point_step = info.point_step as usize;
    if output.len() != total_points.saturating_mul(point_step) {
        return Err("Decoded CloudINI output buffer has the wrong size".to_string());
    }
    let regular_fields = regular_fields(info);
    let mut regular_decoders = build_decoders(&regular_fields, info.encoding_opt, info.version);
    let adaptive_fields = adaptive_fields(info);
    let mut output_point = 0;
    let mut decompressor = ChunkDecompressor::new(info.compression_opt)?;

    if info.version < 3 {
        let stage = decompressor.decompress(payload, stage_buffer_bound(info, total_points))?;
        decode_stage(
            info,
            stage,
            total_points,
            &mut regular_decoders,
            &adaptive_fields,
            output,
            0,
        )?;
        return Ok(());
    }

    let mut position = 0;
    while position < payload.len() {
        if output_point >= total_points {
            return Err("Encoded data contains more chunks than declared points".to_string());
        }
        let chunk_len = read_u32(payload, &mut position, "chunk size")? as usize;
        let chunk = read_slice(payload, &mut position, chunk_len, "chunk data")?;
        let chunk_points = (total_points - output_point).min(POINTS_PER_CHUNK);
        let stage = decompressor.decompress(chunk, stage_buffer_bound(info, chunk_points))?;
        decode_stage(
            info,
            stage,
            chunk_points,
            &mut regular_decoders,
            &adaptive_fields,
            output,
            output_point,
        )?;
        output_point += chunk_points;
    }
    if output_point != total_points {
        return Err("Encoded data ended before all declared points were decoded".to_string());
    }
    Ok(())
}

fn decode_stage(
    info: &EncodingInfo,
    stage: &[u8],
    point_count: usize,
    regular_decoders: &mut [FieldDecoder],
    adaptive_fields: &[AdaptiveField],
    output: &mut [u8],
    first_output_point: usize,
) -> Result<()> {
    for decoder in regular_decoders.iter_mut() {
        decoder.reset();
    }

    let point_step = info.point_step as usize;
    let mut position = 0;
    let chunk_output_start = first_output_point * point_step;
    let chunk_output_end = chunk_output_start + point_count * point_step;
    let used_fast_path = if let [decoder] = regular_decoders {
        decoder.decode_points(
            stage,
            &mut position,
            &mut output[chunk_output_start..chunk_output_end],
            point_step,
        )?
    } else {
        decode_multi_points(
            regular_decoders,
            stage,
            &mut position,
            &mut output[chunk_output_start..chunk_output_end],
            point_step,
        )?
    };
    if !used_fast_path {
        for point_index in 0..point_count {
            let output_start = (first_output_point + point_index) * point_step;
            let point = &mut output[output_start..output_start + point_step];
            for decoder in regular_decoders.iter_mut() {
                decoder.decode(stage, &mut position, point)?;
            }
        }
    }

    for field in adaptive_fields {
        decode_adaptive_section(
            field,
            stage,
            &mut position,
            point_count,
            point_step,
            first_output_point,
            output,
        )?;
    }
    if position != stage.len() {
        return Err("CloudINI chunk has trailing bytes after decode".to_string());
    }
    Ok(())
}

fn uses_v5_codec(info: &EncodingInfo) -> bool {
    info.version >= 5
        && info.encoding_opt == EncodingOptions::Lossy
        && !adaptive_fields(info).is_empty()
}

fn leading_lossy_float_count(info: &EncodingInfo) -> usize {
    if info.encoding_opt != EncodingOptions::Lossy {
        return 0;
    }
    let count = info
        .fields
        .iter()
        .take_while(|field| field.field_type == FieldType::Float32 && field.resolution.is_some())
        .count();
    if count == 3 || count == 4 { count } else { 0 }
}

fn is_adaptive_integer(field_type: FieldType) -> bool {
    matches!(
        field_type,
        FieldType::Int16
            | FieldType::Uint16
            | FieldType::Int32
            | FieldType::Uint32
            | FieldType::Int64
            | FieldType::Uint64
    )
}

fn adaptive_fields(info: &EncodingInfo) -> Vec<AdaptiveField> {
    if info.version < 5 || info.encoding_opt != EncodingOptions::Lossy {
        return Vec::new();
    }
    info.fields[leading_lossy_float_count(info)..]
        .iter()
        .filter(|field| is_adaptive_integer(field.field_type))
        .map(|field| AdaptiveField {
            offset: field.offset as usize,
            field_type: field.field_type,
            size: field.field_type.size_of(),
            committed_mode: None,
        })
        .collect()
}

fn regular_fields(info: &EncodingInfo) -> Vec<PointField> {
    if !uses_v5_codec_without_recursion(info) {
        return info.fields.clone();
    }
    let leading = leading_lossy_float_count(info);
    info.fields
        .iter()
        .enumerate()
        .filter(|(index, field)| *index < leading || !is_adaptive_integer(field.field_type))
        .map(|(_, field)| field.clone())
        .collect()
}

fn uses_v5_codec_without_recursion(info: &EncodingInfo) -> bool {
    if info.version < 5 || info.encoding_opt != EncodingOptions::Lossy {
        return false;
    }
    info.fields[leading_lossy_float_count(info)..]
        .iter()
        .any(|field| is_adaptive_integer(field.field_type))
}

fn append_regular_fields(encoders: &mut [FieldEncoder], point: &[u8], output: &mut Vec<u8>) {
    for encoder in encoders {
        encoder.encode(point, output);
    }
}

fn append_regular_chunk(
    encoders: &mut [FieldEncoder],
    cloud_data: &[u8],
    point_step: usize,
    output: &mut Vec<u8>,
) {
    if let [encoder] = encoders
        && encoder.encode_points(cloud_data, point_step, output)
    {
        return;
    }
    if encode_multi_points(encoders, cloud_data, point_step, output) {
        return;
    }
    for point in cloud_data.chunks_exact(point_step) {
        append_regular_fields(encoders, point, output);
    }
}

fn append_streaming_adaptive_section(
    field: &AdaptiveField,
    cloud_data: &[u8],
    point_step: usize,
    output: &mut Vec<u8>,
) -> Result<()> {
    let mode = field
        .committed_mode
        .expect("streaming adaptive fields always have a committed mode");
    match field.field_type {
        FieldType::Int16 => {
            append_streaming_adaptive_kind::<3>(mode, field, cloud_data, point_step, output)?
        }
        FieldType::Uint16 => {
            append_streaming_adaptive_kind::<4>(mode, field, cloud_data, point_step, output)?
        }
        FieldType::Int32 => {
            append_streaming_adaptive_kind::<5>(mode, field, cloud_data, point_step, output)?
        }
        FieldType::Uint32 => {
            append_streaming_adaptive_kind::<6>(mode, field, cloud_data, point_step, output)?
        }
        FieldType::Int64 => {
            append_streaming_adaptive_kind::<9>(mode, field, cloud_data, point_step, output)?
        }
        FieldType::Uint64 => {
            append_streaming_adaptive_kind::<10>(mode, field, cloud_data, point_step, output)?
        }
        _ => unreachable!("adaptive fields are always 16, 32, or 64 bit integers"),
    }
    Ok(())
}

#[inline(always)]
fn append_streaming_adaptive_kind<const KIND: u8>(
    mode: AdaptiveMode,
    field: &AdaptiveField,
    cloud_data: &[u8],
    point_step: usize,
    output: &mut Vec<u8>,
) -> Result<()> {
    output.push(mode as u8);
    match mode {
        AdaptiveMode::DeltaVarint => {
            let mut previous = 0_i64;
            if KIND == 6 && point_step == 4 && field.offset == 0 {
                encode_dense_u32_points(cloud_data, output, &mut previous);
            } else {
                for point in cloud_data.chunks_exact(point_step) {
                    let value = read_integer_kind::<KIND>(point, field.offset);
                    append_varint(value.wrapping_sub(previous), output);
                    previous = value;
                }
            }
        }
        AdaptiveMode::Palette => {
            output.pop();
            let values = cloud_data
                .chunks_exact(point_step)
                .map(|point| read_integer_kind::<KIND>(point, field.offset))
                .collect::<Vec<_>>();
            append_adaptive_section(mode, &values, field.size, output)?;
        }
        AdaptiveMode::Rle => {
            let count_position = output.len();
            output.extend_from_slice(&0_u32.to_le_bytes());
            let mut current = None;
            let mut run_length = 0;
            let mut run_count = 0_u32;
            for point in cloud_data.chunks_exact(point_step) {
                let value = read_integer_kind::<KIND>(point, field.offset);
                if current == Some(value) {
                    run_length += 1;
                } else {
                    append_rle_run(output, current, run_length, field.size);
                    run_count += u32::from(current.is_some());
                    current = Some(value);
                    run_length = 1;
                }
            }
            append_rle_run(output, current, run_length, field.size);
            run_count += u32::from(current.is_some());
            output[count_position..count_position + 4].copy_from_slice(&run_count.to_le_bytes());
        }
        AdaptiveMode::DeltaRle => {
            let count_position = output.len();
            output.extend_from_slice(&0_u32.to_le_bytes());
            let mut previous_value = 0_i64;
            let mut current_delta = None;
            let mut run_length = 0;
            let mut run_count = 0_u32;
            for point in cloud_data.chunks_exact(point_step) {
                let value = read_integer_kind::<KIND>(point, field.offset);
                let delta = value.wrapping_sub(previous_value);
                previous_value = value;
                if current_delta == Some(delta) {
                    run_length += 1;
                } else {
                    append_delta_rle_run(output, current_delta, run_length);
                    run_count += u32::from(current_delta.is_some());
                    current_delta = Some(delta);
                    run_length = 1;
                }
            }
            append_delta_rle_run(output, current_delta, run_length);
            run_count += u32::from(current_delta.is_some());
            output[count_position..count_position + 4].copy_from_slice(&run_count.to_le_bytes());
        }
    }
    Ok(())
}

#[inline(always)]
fn append_rle_run(output: &mut Vec<u8>, value: Option<i64>, length: usize, value_size: usize) {
    if let Some(value) = value {
        output.extend_from_slice(&(value as u64).to_le_bytes()[..value_size]);
        append_uvarint(length as u64, output);
    }
}

#[inline(always)]
fn append_delta_rle_run(output: &mut Vec<u8>, delta: Option<i64>, length: usize) {
    if let Some(delta) = delta {
        append_varint(delta, output);
        append_uvarint(length as u64, output);
    }
}

fn select_mode(values: &[i64], value_size: usize) -> AdaptiveMode {
    let delta_size = delta_varint_size(values);
    let palette_size = palette_size(values, value_size);
    let rle_size = rle_size(values, value_size);
    let delta_rle_size = delta_rle_size(values);
    let mut best_mode = AdaptiveMode::DeltaVarint;
    let mut best_size = delta_size;
    for (mode, size) in [
        (AdaptiveMode::Palette, palette_size),
        (AdaptiveMode::Rle, rle_size),
        (AdaptiveMode::DeltaRle, delta_rle_size),
    ] {
        if size < best_size {
            best_mode = mode;
            best_size = size;
        }
    }
    best_mode
}

fn append_adaptive_section(
    mode: AdaptiveMode,
    values: &[i64],
    value_size: usize,
    output: &mut Vec<u8>,
) -> Result<()> {
    output.push(mode as u8);
    match mode {
        AdaptiveMode::DeltaVarint => {
            let mut previous = 0_i64;
            for value in values {
                append_varint(value.wrapping_sub(previous), output);
                previous = *value;
            }
        }
        AdaptiveMode::Palette => {
            let (palette, indexes) = make_palette(values);
            let palette_len = u16::try_from(palette.len())
                .map_err(|_| "V5 palette contains more than 65535 values".to_string())?;
            output.extend_from_slice(&palette_len.to_le_bytes());
            for value in palette {
                output.extend_from_slice(&(value as u64).to_le_bytes()[..value_size]);
            }
            append_bitpacked_indexes(
                &indexes,
                bits_for_palette(indexes.iter().max().copied()),
                output,
            );
        }
        AdaptiveMode::Rle => {
            output.extend_from_slice(&(count_value_runs(values) as u32).to_le_bytes());
            let mut current = None;
            let mut run_length = 0;
            for &value in values {
                if current == Some(value) {
                    run_length += 1;
                } else {
                    append_rle_run(output, current, run_length, value_size);
                    current = Some(value);
                    run_length = 1;
                }
            }
            append_rle_run(output, current, run_length, value_size);
        }
        AdaptiveMode::DeltaRle => {
            output.extend_from_slice(&(count_delta_runs(values) as u32).to_le_bytes());
            let mut previous_value = 0_i64;
            let mut current_delta = None;
            let mut run_length = 0;
            for &value in values {
                let delta = value.wrapping_sub(previous_value);
                previous_value = value;
                if current_delta == Some(delta) {
                    run_length += 1;
                } else {
                    append_delta_rle_run(output, current_delta, run_length);
                    current_delta = Some(delta);
                    run_length = 1;
                }
            }
            append_delta_rle_run(output, current_delta, run_length);
        }
    }
    Ok(())
}

fn decode_adaptive_section(
    field: &AdaptiveField,
    stage: &[u8],
    position: &mut usize,
    point_count: usize,
    point_step: usize,
    first_output_point: usize,
    output: &mut [u8],
) -> Result<()> {
    let mode = match read_u8(stage, position, "V5 adaptive mode")? {
        0 => AdaptiveMode::DeltaVarint,
        1 => AdaptiveMode::Palette,
        2 => AdaptiveMode::Rle,
        3 => AdaptiveMode::DeltaRle,
        value => return Err(format!("V5 adaptive int: unknown mode byte {value}")),
    };
    match mode {
        AdaptiveMode::DeltaVarint => match field.size {
            2 => decode_adaptive_delta::<2>(
                stage,
                position,
                point_count,
                point_step,
                first_output_point,
                field.offset,
                output,
            )?,
            4 => decode_adaptive_delta::<4>(
                stage,
                position,
                point_count,
                point_step,
                first_output_point,
                field.offset,
                output,
            )?,
            8 => decode_adaptive_delta::<8>(
                stage,
                position,
                point_count,
                point_step,
                first_output_point,
                field.offset,
                output,
            )?,
            _ => unreachable!("adaptive fields are always 16, 32, or 64 bit integers"),
        },
        AdaptiveMode::Palette => {
            let palette_count = usize::from(read_u16(stage, position, "V5 palette count")?);
            if palette_count == 0 {
                return Err("V5 adaptive int: empty palette".to_string());
            }
            let mut palette = Vec::with_capacity(palette_count);
            for _ in 0..palette_count {
                palette.push(read_raw(read_slice(
                    stage,
                    position,
                    field.size,
                    "V5 palette value",
                )?));
            }
            let bits = bits_for_palette(Some((palette_count - 1) as u32));
            let index_bytes = (usize::from(bits) * point_count).div_ceil(8);
            let packed = read_slice(stage, position, index_bytes, "V5 palette indexes")?;
            match field.size {
                2 => decode_palette::<2>(
                    packed,
                    point_count,
                    bits,
                    &palette,
                    point_step,
                    first_output_point,
                    field.offset,
                    output,
                )?,
                4 => decode_palette::<4>(
                    packed,
                    point_count,
                    bits,
                    &palette,
                    point_step,
                    first_output_point,
                    field.offset,
                    output,
                )?,
                8 => decode_palette::<8>(
                    packed,
                    point_count,
                    bits,
                    &palette,
                    point_step,
                    first_output_point,
                    field.offset,
                    output,
                )?,
                _ => unreachable!("adaptive fields are always 16, 32, or 64 bit integers"),
            }
        }
        AdaptiveMode::Rle => match field.size {
            2 => decode_rle::<2>(
                stage,
                position,
                point_count,
                point_step,
                first_output_point,
                field.offset,
                output,
            )?,
            4 => decode_rle::<4>(
                stage,
                position,
                point_count,
                point_step,
                first_output_point,
                field.offset,
                output,
            )?,
            8 => decode_rle::<8>(
                stage,
                position,
                point_count,
                point_step,
                first_output_point,
                field.offset,
                output,
            )?,
            _ => unreachable!("adaptive fields are always 16, 32, or 64 bit integers"),
        },
        AdaptiveMode::DeltaRle => match field.size {
            2 => decode_delta_rle::<2>(
                stage,
                position,
                point_count,
                point_step,
                first_output_point,
                field.offset,
                output,
            )?,
            4 => decode_delta_rle::<4>(
                stage,
                position,
                point_count,
                point_step,
                first_output_point,
                field.offset,
                output,
            )?,
            8 => decode_delta_rle::<8>(
                stage,
                position,
                point_count,
                point_step,
                first_output_point,
                field.offset,
                output,
            )?,
            _ => unreachable!("adaptive fields are always 16, 32, or 64 bit integers"),
        },
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_palette<const SIZE: usize>(
    packed: &[u8],
    point_count: usize,
    bits: u8,
    palette: &[u64],
    point_step: usize,
    first_output_point: usize,
    field_offset: usize,
    output: &mut [u8],
) -> Result<()> {
    if bits == 0 {
        let value = palette[0].to_le_bytes();
        for point_index in 0..point_count {
            let output_start = (first_output_point + point_index) * point_step + field_offset;
            output[output_start..output_start + SIZE].copy_from_slice(&value[..SIZE]);
        }
        return Ok(());
    }

    let mut packed_position = 0;
    let mut scratch = 0_u64;
    let mut held = 0_u8;
    let mask = (1_u64 << bits) - 1;
    for point_index in 0..point_count {
        while held < bits {
            scratch |= u64::from(packed[packed_position]) << held;
            packed_position += 1;
            held += 8;
        }
        let index = (scratch & mask) as usize;
        scratch >>= bits;
        held -= bits;
        let value = palette
            .get(index)
            .ok_or_else(|| "V5 adaptive int: palette index out of range".to_string())?
            .to_le_bytes();
        let output_start = (first_output_point + point_index) * point_step + field_offset;
        output[output_start..output_start + SIZE].copy_from_slice(&value[..SIZE]);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_rle<const SIZE: usize>(
    stage: &[u8],
    position: &mut usize,
    point_count: usize,
    point_step: usize,
    first_output_point: usize,
    field_offset: usize,
    output: &mut [u8],
) -> Result<()> {
    let run_count = read_u32(stage, position, "V5 RLE run count")?;
    let mut point_index = 0;
    for _ in 0..run_count {
        let value = read_slice(stage, position, SIZE, "V5 RLE value")?;
        let run_length = read_uvarint(stage, position)? as usize;
        if run_length > point_count - point_index {
            return Err("V5 adaptive int: RLE run exceeds point count".to_string());
        }
        for _ in 0..run_length {
            let output_start = (first_output_point + point_index) * point_step + field_offset;
            output[output_start..output_start + SIZE].copy_from_slice(value);
            point_index += 1;
        }
    }
    if point_index != point_count {
        return Err("V5 adaptive int: RLE runs do not fill chunk".to_string());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn decode_delta_rle<const SIZE: usize>(
    stage: &[u8],
    position: &mut usize,
    point_count: usize,
    point_step: usize,
    first_output_point: usize,
    field_offset: usize,
    output: &mut [u8],
) -> Result<()> {
    if SIZE == 4 && point_step == 4 && field_offset == 0 {
        return decode_delta_rle_dense_u32(
            stage,
            position,
            point_count,
            first_output_point,
            output,
        );
    }

    let run_count = read_u32(stage, position, "V5 delta-RLE run count")?;
    let mut point_index = 0;
    let mut previous = 0_i64;
    for _ in 0..run_count {
        let delta = decode_varint_at(stage, position)?;
        let run_length = read_uvarint(stage, position)? as usize;
        if run_length > point_count - point_index {
            return Err("V5 adaptive int: delta-RLE run exceeds point count".to_string());
        }
        for _ in 0..run_length {
            previous = previous.wrapping_add(delta);
            let output_start = (first_output_point + point_index) * point_step + field_offset;
            output[output_start..output_start + SIZE]
                .copy_from_slice(&previous.to_le_bytes()[..SIZE]);
            point_index += 1;
        }
    }
    if point_index != point_count {
        return Err("V5 adaptive int: delta-RLE runs do not fill chunk".to_string());
    }
    Ok(())
}

fn decode_delta_rle_dense_u32(
    stage: &[u8],
    position: &mut usize,
    point_count: usize,
    first_output_point: usize,
    output: &mut [u8],
) -> Result<()> {
    let run_count = read_u32(stage, position, "V5 delta-RLE run count")?;
    let mut point_index = 0;
    let mut previous = 0_u32;
    for _ in 0..run_count {
        let delta = decode_varint_at(stage, position)? as u32;
        let delta_2 = delta.wrapping_mul(2);
        let delta_3 = delta.wrapping_mul(3);
        let delta_4 = delta.wrapping_mul(4);
        let run_length = read_uvarint(stage, position)? as usize;
        if run_length > point_count - point_index {
            return Err("V5 adaptive int: delta-RLE run exceeds point count".to_string());
        }
        let start = (first_output_point + point_index) * 4;
        let end = start + run_length * 4;
        let destination = &mut output[start..end];
        let mut groups = destination.chunks_exact_mut(16);
        for group in &mut groups {
            let value_0 = previous.wrapping_add(delta);
            let value_1 = previous.wrapping_add(delta_2);
            let value_2 = previous.wrapping_add(delta_3);
            let value_3 = previous.wrapping_add(delta_4);
            let values = [
                value_0.to_le(),
                value_1.to_le(),
                value_2.to_le(),
                value_3.to_le(),
            ];
            // Every exact chunk has space for four unaligned u32 values.
            unsafe {
                group
                    .as_mut_ptr()
                    .cast::<[u32; 4]>()
                    .write_unaligned(values);
            }
            previous = value_3;
        }
        for bytes in groups.into_remainder().chunks_exact_mut(4) {
            previous = previous.wrapping_add(delta);
            bytes.copy_from_slice(&previous.to_le_bytes());
        }
        point_index += run_length;
    }
    if point_index != point_count {
        return Err("V5 adaptive int: delta-RLE runs do not fill chunk".to_string());
    }
    Ok(())
}

#[inline(never)]
fn decode_adaptive_delta<const SIZE: usize>(
    stage: &[u8],
    position: &mut usize,
    point_count: usize,
    point_step: usize,
    first_output_point: usize,
    field_offset: usize,
    output: &mut [u8],
) -> Result<()> {
    if SIZE == 4 && point_step == 4 && field_offset == 0 {
        return decode_adaptive_delta_dense_u32(
            stage,
            position,
            point_count,
            first_output_point,
            output,
        );
    }

    let mut previous = 0_i64;
    for point_index in 0..point_count {
        previous = previous.wrapping_add(decode_varint_at(stage, position)?);
        let output_start = (first_output_point + point_index) * point_step + field_offset;
        output[output_start..output_start + SIZE].copy_from_slice(&previous.to_le_bytes()[..SIZE]);
    }
    Ok(())
}

fn decode_adaptive_delta_dense_u32(
    stage: &[u8],
    position: &mut usize,
    point_count: usize,
    first_output_point: usize,
    output: &mut [u8],
) -> Result<()> {
    let mut cursor = *position;
    let mut previous = 0_u32;
    let start = first_output_point * 4;
    let end = start + point_count * 4;
    for bytes in output[start..end].chunks_exact_mut(4) {
        previous = previous.wrapping_add(decode_varint_at(stage, &mut cursor)? as u32);
        bytes.copy_from_slice(&previous.to_le_bytes());
    }
    *position = cursor;
    Ok(())
}

fn delta_varint_size(values: &[i64]) -> usize {
    let mut size = 1;
    let mut previous = 0_i64;
    for value in values {
        size += varint_size(value.wrapping_sub(previous));
        previous = *value;
    }
    size
}

fn palette_size(values: &[i64], value_size: usize) -> usize {
    let (palette, _) = make_palette(values);
    let bits = bits_for_palette(
        palette
            .len()
            .checked_sub(1)
            .and_then(|value| u32::try_from(value).ok()),
    );
    1 + 2 + palette.len() * value_size + (usize::from(bits) * values.len()).div_ceil(8)
}

fn rle_size(values: &[i64], value_size: usize) -> usize {
    let mut size = 1 + 4;
    let mut current = None;
    let mut run_length = 0;
    for &value in values {
        if current == Some(value) {
            run_length += 1;
        } else {
            if current.is_some() {
                size += value_size + uvarint_size(run_length as u64);
            }
            current = Some(value);
            run_length = 1;
        }
    }
    if current.is_some() {
        size += value_size + uvarint_size(run_length as u64);
    }
    size
}

fn delta_rle_size(values: &[i64]) -> usize {
    let mut size = 1 + 4;
    let mut previous_value = 0_i64;
    let mut current_delta = None;
    let mut run_length = 0;
    for &value in values {
        let delta = value.wrapping_sub(previous_value);
        previous_value = value;
        if current_delta == Some(delta) {
            run_length += 1;
        } else {
            if let Some(current_delta) = current_delta {
                size += varint_size(current_delta) + uvarint_size(run_length as u64);
            }
            current_delta = Some(delta);
            run_length = 1;
        }
    }
    if let Some(current_delta) = current_delta {
        size += varint_size(current_delta) + uvarint_size(run_length as u64);
    }
    size
}

fn make_palette(values: &[i64]) -> (Vec<i64>, Vec<u32>) {
    let mut lookup = HashMap::new();
    let mut palette = Vec::new();
    let mut indexes = Vec::with_capacity(values.len());
    for value in values {
        let index = match lookup.get(value) {
            Some(index) => *index,
            None => {
                let index = palette.len() as u32;
                palette.push(*value);
                lookup.insert(*value, index);
                index
            }
        };
        indexes.push(index);
    }
    (palette, indexes)
}

fn count_value_runs(values: &[i64]) -> usize {
    values
        .windows(2)
        .filter(|values| values[0] != values[1])
        .count()
        + usize::from(!values.is_empty())
}

fn count_delta_runs(values: &[i64]) -> usize {
    let mut run_count = 0;
    let mut previous = 0_i64;
    let mut previous_delta = None;
    for &value in values {
        let delta = value.wrapping_sub(previous);
        previous = value;
        if previous_delta != Some(delta) {
            run_count += 1;
            previous_delta = Some(delta);
        }
    }
    run_count
}

fn varint_size(value: i64) -> usize {
    let mut buffer = [0; 10];
    encode_varint64(value, &mut buffer)
}

fn append_uvarint(mut value: u64, output: &mut Vec<u8>) {
    while value > 0x7f {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn read_uvarint(data: &[u8], position: &mut usize) -> Result<u64> {
    let mut value = 0_u64;
    let mut shift = 0_u32;
    loop {
        let byte = read_u8(data, position, "V5 unsigned varint")?;
        if shift >= 64 {
            return Err("V5 adaptive int: unsigned varint overflow".to_string());
        }
        if shift == 63 && byte & 0x7f > 1 {
            return Err("V5 adaptive int: unsigned varint overflow".to_string());
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
}

fn uvarint_size(mut value: u64) -> usize {
    let mut size = 1;
    while value > 0x7f {
        value >>= 7;
        size += 1;
    }
    size
}

fn bits_for_palette(max_index: Option<u32>) -> u8 {
    match max_index {
        None | Some(0) => 0,
        Some(value) => (u32::BITS - value.leading_zeros()) as u8,
    }
}

fn append_bitpacked_indexes(indexes: &[u32], bits: u8, output: &mut Vec<u8>) {
    if bits == 0 {
        return;
    }
    let mut scratch = 0_u64;
    let mut held = 0_u8;
    for index in indexes {
        scratch |= u64::from(*index) << held;
        held += bits;
        while held >= 8 {
            output.push(scratch as u8);
            scratch >>= 8;
            held -= 8;
        }
    }
    if held > 0 {
        output.push(scratch as u8);
    }
}

fn read_integer(bytes: &[u8], field_type: FieldType) -> i64 {
    match field_type {
        FieldType::Int16 => i16::from_le_bytes(bytes.try_into().unwrap()) as i64,
        FieldType::Uint16 => u16::from_le_bytes(bytes.try_into().unwrap()) as i64,
        FieldType::Int32 => i32::from_le_bytes(bytes.try_into().unwrap()) as i64,
        FieldType::Uint32 => u32::from_le_bytes(bytes.try_into().unwrap()) as i64,
        FieldType::Int64 => i64::from_le_bytes(bytes.try_into().unwrap()),
        FieldType::Uint64 => u64::from_le_bytes(bytes.try_into().unwrap()) as i64,
        _ => unreachable!("adaptive fields are always 16/32/64-bit integers"),
    }
}

fn read_raw(bytes: &[u8]) -> u64 {
    let mut buffer = [0; 8];
    buffer[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(buffer)
}

struct ChunkCompressor {
    option: CompressionOption,
    zstd: Option<zstd::bulk::Compressor<'static>>,
    buffer: Vec<u8>,
}

impl ChunkCompressor {
    fn new(option: CompressionOption) -> Result<Self> {
        let zstd = (option == CompressionOption::Zstd)
            .then(|| zstd::bulk::Compressor::new(1))
            .transpose()
            .map_err(|error| format!("Failed to initialize ZSTD compressor: {error}"))?;
        Ok(Self {
            option,
            zstd,
            buffer: Vec::new(),
        })
    }

    fn compress<'a>(&'a mut self, data: &'a [u8]) -> Result<&'a [u8]> {
        match self.option {
            CompressionOption::None => Ok(data),
            CompressionOption::Lz4 => {
                let capacity = lz4_flex::block::get_maximum_output_size(data.len());
                self.buffer.resize(capacity, 0);
                let size = lz4_flex::block::compress_into(data, &mut self.buffer)
                    .map_err(|error| format!("LZ4 compression failed: {error}"))?;
                Ok(&self.buffer[..size])
            }
            CompressionOption::Zstd => {
                self.buffer.clear();
                self.buffer
                    .reserve(zstd::zstd_safe::compress_bound(data.len()));
                self.zstd
                    .as_mut()
                    .expect("ZSTD compressor is initialized for ZSTD compression")
                    .compress_to_buffer(data, &mut self.buffer)
                    .map_err(|error| format!("ZSTD compression failed: {error}"))?;
                Ok(&self.buffer)
            }
        }
    }
}

struct ChunkDecompressor {
    option: CompressionOption,
    zstd: Option<zstd::bulk::Decompressor<'static>>,
    buffer: Vec<u8>,
}

impl ChunkDecompressor {
    fn new(option: CompressionOption) -> Result<Self> {
        let zstd = (option == CompressionOption::Zstd)
            .then(zstd::bulk::Decompressor::new)
            .transpose()
            .map_err(|error| format!("Failed to initialize ZSTD decompressor: {error}"))?;
        Ok(Self {
            option,
            zstd,
            buffer: Vec::new(),
        })
    }

    fn decompress<'a>(&'a mut self, data: &'a [u8], max_output: usize) -> Result<&'a [u8]> {
        match self.option {
            CompressionOption::None => Ok(data),
            CompressionOption::Lz4 => {
                self.buffer.resize(max_output, 0);
                let size = lz4_flex::block::decompress_into(data, &mut self.buffer)
                    .map_err(|error| format!("LZ4 decompression failed: {error}"))?;
                Ok(&self.buffer[..size])
            }
            CompressionOption::Zstd => {
                self.buffer.clear();
                self.buffer.reserve(max_output);
                self.zstd
                    .as_mut()
                    .expect("ZSTD decompressor is initialized for ZSTD compression")
                    .decompress_to_buffer(data, &mut self.buffer)
                    .map_err(|error| format!("ZSTD decompression failed: {error}"))?;
                Ok(&self.buffer)
            }
        }
    }
}

fn stage_buffer_bound(info: &EncodingInfo, point_count: usize) -> usize {
    let max_per_point = info
        .fields
        .iter()
        .map(|field| match field.field_type {
            FieldType::Int8 | FieldType::Uint8 => 1,
            FieldType::Int16
            | FieldType::Uint16
            | FieldType::Int32
            | FieldType::Uint32
            | FieldType::Int64
            | FieldType::Uint64 => 10,
            FieldType::Float32
                if info.encoding_opt == EncodingOptions::Lossy && field.resolution.is_some() =>
            {
                10
            }
            FieldType::Float32 => 7,
            FieldType::Float64
                if info.encoding_opt == EncodingOptions::Lossy && field.resolution.is_some() =>
            {
                10
            }
            FieldType::Float64 => 11,
            FieldType::Unknown => 0,
        })
        .sum::<usize>()
        .max(info.point_step as usize);
    point_count * (max_per_point + 64) + info.fields.len() * 64 + 1024
}

fn validate_info(info: &EncodingInfo) -> Result<()> {
    if info.point_step == 0 {
        return Err("point_step cannot be 0".to_string());
    }
    if !(2..=ENCODING_VERSION).contains(&info.version) {
        return Err(format!(
            "Unsupported encoding version. Current is: {ENCODING_VERSION}, got: {}",
            info.version
        ));
    }
    for field in &info.fields {
        let size = field.field_type.size_of();
        if size == 0 {
            return Err(format!("Unsupported field type for '{}'", field.name));
        }
        if field.offset as usize + size > info.point_step as usize {
            return Err(format!(
                "Field '{}' exceeds point_step {}",
                field.name, info.point_step
            ));
        }
        if let Some(resolution) = field.resolution
            && (!resolution.is_finite() || resolution <= 0.0)
        {
            return Err(format!(
                "Field '{}' resolution must be finite and positive",
                field.name
            ));
        }
        if info.encoding_opt == EncodingOptions::Lossy
            && field.field_type == FieldType::Float32
            && field
                .resolution
                .is_some_and(|resolution| !resolution.recip().is_finite())
        {
            return Err(format!(
                "Field '{}' resolution is too small for FLOAT32 lossy encoding",
                field.name
            ));
        }
    }
    Ok(())
}

fn validate_integer_varint_deltas(
    info: &EncodingInfo,
    cloud_data: &[u8],
    point_step: usize,
) -> Result<()> {
    if info.encoding_opt == EncodingOptions::None {
        return Ok(());
    }
    for field in &info.fields {
        if !matches!(field.field_type, FieldType::Int64 | FieldType::Uint64) {
            continue;
        }
        let mut previous = 0_i64;
        for point in cloud_data.chunks_exact(point_step) {
            let value = read_integer(
                &point[field.offset as usize..field.offset as usize + 8],
                field.field_type,
            );
            if value.wrapping_sub(previous) == i64::MIN {
                return Err(format!(
                    "Field '{}' contains an integer delta that CloudINI varint cannot represent",
                    field.name
                ));
            }
            previous = value;
        }
    }
    Ok(())
}

fn split_yaml_entry(line: &str) -> Result<(&str, &str)> {
    let (key, value) = line
        .split_once(':')
        .ok_or_else(|| format!("Malformed CloudINI YAML line '{line}'"))?;
    Ok((key.trim(), value.trim()))
}

fn parse_yaml_number<T>(value: &str, key: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|error| format!("Invalid {key} '{value}': {error}"))
}

fn apply_field_yaml(field: &mut PointField, entry: (&str, &str)) -> Result<()> {
    let (key, value) = entry;
    match key {
        "name" => field.name = value.to_string(),
        "offset" => field.offset = parse_yaml_number(value, key)?,
        "type" => {
            field.field_type = FieldType::from_str(value)
                .ok_or_else(|| format!("Invalid field type '{value}'"))?;
        }
        "resolution" => {
            field.resolution = if value.eq_ignore_ascii_case("null") {
                None
            } else {
                Some(parse_yaml_number(value, key)?)
            };
        }
        _ => {}
    }
    Ok(())
}

fn write_magic(version: u8, output: &mut Vec<u8>) {
    output.extend_from_slice(MAGIC_HEADER);
    output.push(b'0' + version / 10);
    output.push(b'0' + version % 10);
}

fn encoding_option(value: u8) -> Result<EncodingOptions> {
    match value {
        0 => Ok(EncodingOptions::None),
        1 => Ok(EncodingOptions::Lossy),
        2 => Ok(EncodingOptions::Lossless),
        _ => Err(format!("Invalid CloudINI encoding option {value}")),
    }
}

fn compression_option(value: u8) -> Result<CompressionOption> {
    match value {
        0 => Ok(CompressionOption::None),
        1 => Ok(CompressionOption::Lz4),
        2 => Ok(CompressionOption::Zstd),
        _ => Err(format!("Invalid CloudINI compression option {value}")),
    }
}

fn field_type(value: u8) -> Result<FieldType> {
    match value {
        0 => Ok(FieldType::Unknown),
        1 => Ok(FieldType::Int8),
        2 => Ok(FieldType::Uint8),
        3 => Ok(FieldType::Int16),
        4 => Ok(FieldType::Uint16),
        5 => Ok(FieldType::Int32),
        6 => Ok(FieldType::Uint32),
        7 => Ok(FieldType::Float32),
        8 => Ok(FieldType::Float64),
        9 => Ok(FieldType::Int64),
        10 => Ok(FieldType::Uint64),
        _ => Err(format!("Invalid CloudINI field type {value}")),
    }
}

fn read_u8(data: &[u8], position: &mut usize, context: &str) -> Result<u8> {
    let value = *data
        .get(*position)
        .ok_or_else(|| format!("Truncated CloudINI data while reading {context}"))?;
    *position += 1;
    Ok(value)
}

fn read_u16(data: &[u8], position: &mut usize, context: &str) -> Result<u16> {
    Ok(u16::from_le_bytes(read_array(data, position, context)?))
}

fn read_u32(data: &[u8], position: &mut usize, context: &str) -> Result<u32> {
    Ok(u32::from_le_bytes(read_array(data, position, context)?))
}

fn read_array<const N: usize>(data: &[u8], position: &mut usize, context: &str) -> Result<[u8; N]> {
    read_slice(data, position, N, context)?
        .try_into()
        .map_err(|_| format!("Truncated CloudINI data while reading {context}"))
}

fn read_slice<'a>(
    data: &'a [u8],
    position: &mut usize,
    length: usize,
    context: &str,
) -> Result<&'a [u8]> {
    let end = position
        .checked_add(length)
        .ok_or_else(|| format!("CloudINI length overflow while reading {context}"))?;
    let value = data
        .get(*position..end)
        .ok_or_else(|| format!("Truncated CloudINI data while reading {context}"))?;
    *position = end;
    Ok(value)
}
