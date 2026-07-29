mod codec;
mod field_codec;
mod preprocess;
mod types;
mod varint;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList, PyModule, PySlice};
use std::sync::OnceLock;

use crate::types::{
    CompressionOption, EncodingInfo, EncodingOptions, FieldType, POINTS_PER_CHUNK, PointField,
};

#[pyclass(name = "PointField", module = "pureini", skip_from_py_object)]
#[derive(Clone)]
struct PyPointField {
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    offset: u32,
    type_value: u8,
    #[pyo3(get, set)]
    resolution: Option<f32>,
}

#[pymethods]
impl PyPointField {
    #[new]
    #[pyo3(signature = (name, offset=0, r#type=0, resolution=None))]
    fn new(name: String, offset: u32, r#type: u8, resolution: Option<f32>) -> PyResult<Self> {
        field_type(r#type)?;
        Ok(Self {
            name,
            offset,
            type_value: r#type,
            resolution,
        })
    }

    #[getter]
    fn r#type(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_member(py, "FieldType", self.type_value)
    }

    #[setter]
    fn set_type(&mut self, value: u8) -> PyResult<()> {
        field_type(value)?;
        self.type_value = value;
        Ok(())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let type_repr = enum_member(py, "FieldType", self.type_value)?
            .bind(py)
            .repr()?
            .to_str()?
            .to_string();
        Ok(format!(
            "PointField(name={:?}, offset={}, type={}, resolution={:?})",
            self.name, self.offset, type_repr, self.resolution
        ))
    }

    fn __richcmp__(&self, other: &Self, operation: CompareOp) -> bool {
        let is_equal = self.name == other.name
            && self.offset == other.offset
            && self.type_value == other.type_value
            && self.resolution == other.resolution;
        match operation {
            CompareOp::Eq => is_equal,
            CompareOp::Ne => !is_equal,
            _ => false,
        }
    }
}

impl PyPointField {
    fn to_core(&self) -> PyResult<PointField> {
        Ok(PointField {
            name: self.name.clone(),
            offset: self.offset,
            field_type: field_type(self.type_value)?,
            resolution: self.resolution,
        })
    }

    fn from_core(field: PointField) -> Self {
        Self {
            name: field.name,
            offset: field.offset,
            type_value: field.field_type as u8,
            resolution: field.resolution,
        }
    }
}

#[pyclass(name = "EncodingInfo", module = "pureini")]
struct PyEncodingInfo {
    fields: Py<PyList>,
    #[pyo3(get, set)]
    width: u32,
    #[pyo3(get, set)]
    height: u32,
    #[pyo3(get, set)]
    point_step: u32,
    encoding_value: u8,
    compression_value: u8,
    #[pyo3(get, set)]
    version: u8,
    #[pyo3(get, set)]
    encoding_config: String,
}

#[pymethods]
impl PyEncodingInfo {
    #[new]
    #[pyo3(signature = (
        fields=None,
        width=0,
        height=1,
        point_step=0,
        encoding_opt=1,
        compression_opt=2,
        version=codec::ENCODING_VERSION,
        *,
        encoding_config=String::new()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        fields: Option<&Bound<'_, PyAny>>,
        width: u32,
        height: u32,
        point_step: u32,
        encoding_opt: u8,
        compression_opt: u8,
        version: u8,
        encoding_config: String,
    ) -> PyResult<Self> {
        encoding_option(encoding_opt)?;
        compression_option(compression_opt)?;
        let fields = match fields {
            Some(fields) => iterable_to_list(py, fields)?,
            None => PyList::empty(py).unbind(),
        };
        Ok(Self {
            fields,
            width,
            height,
            point_step,
            encoding_value: encoding_opt,
            compression_value: compression_opt,
            version,
            encoding_config,
        })
    }

    #[getter]
    fn fields(&self, py: Python<'_>) -> Py<PyList> {
        self.fields.clone_ref(py)
    }

    #[setter]
    fn set_fields(&mut self, py: Python<'_>, fields: &Bound<'_, PyAny>) -> PyResult<()> {
        self.fields = iterable_to_list(py, fields)?;
        Ok(())
    }

    #[getter]
    fn encoding_opt(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_member(py, "EncodingOptions", self.encoding_value)
    }

    #[setter]
    fn set_encoding_opt(&mut self, value: u8) -> PyResult<()> {
        encoding_option(value)?;
        self.encoding_value = value;
        Ok(())
    }

    #[getter]
    fn compression_opt(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_member(py, "CompressionOption", self.compression_value)
    }

    #[setter]
    fn set_compression_opt(&mut self, value: u8) -> PyResult<()> {
        compression_option(value)?;
        self.compression_value = value;
        Ok(())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "EncodingInfo(fields={}, width={}, height={}, point_step={}, encoding_opt={}, \
             compression_opt={}, version={})",
            self.fields.bind(py).repr()?.to_str()?,
            self.width,
            self.height,
            self.point_step,
            self.encoding_opt(py)?.bind(py).repr()?.to_str()?,
            self.compression_opt(py)?.bind(py).repr()?.to_str()?,
            self.version,
        ))
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &PyEncodingInfo,
        operation: CompareOp,
    ) -> PyResult<bool> {
        let is_equal = self.to_core(py)? == other.to_core(py)?;
        Ok(match operation {
            CompareOp::Eq => is_equal,
            CompareOp::Ne => !is_equal,
            _ => false,
        })
    }
}

impl PyEncodingInfo {
    fn to_core(&self, py: Python<'_>) -> PyResult<EncodingInfo> {
        let fields = self
            .fields
            .bind(py)
            .iter()
            .map(|field| field.extract::<PyRef<'_, PyPointField>>()?.to_core())
            .collect::<PyResult<Vec<_>>>()?;
        Ok(EncodingInfo {
            fields,
            width: self.width,
            height: self.height,
            point_step: self.point_step,
            encoding_opt: encoding_option(self.encoding_value)?,
            compression_opt: compression_option(self.compression_value)?,
            encoding_config: self.encoding_config.clone(),
            version: self.version,
        })
    }

    fn from_core(py: Python<'_>, info: EncodingInfo) -> PyResult<Self> {
        let fields = PyList::empty(py);
        for field in info.fields {
            fields.append(Py::new(py, PyPointField::from_core(field))?)?;
        }
        Ok(Self {
            fields: fields.unbind(),
            width: info.width,
            height: info.height,
            point_step: info.point_step,
            encoding_value: info.encoding_opt as u8,
            compression_value: info.compression_opt as u8,
            version: info.version,
            encoding_config: info.encoding_config,
        })
    }
}

#[pyclass(name = "PointcloudEncoder", module = "pureini")]
struct PyPointcloudEncoder {
    info: Py<PyEncodingInfo>,
    core_info: EncodingInfo,
    header: OnceLock<Vec<u8>>,
}

#[pymethods]
impl PyPointcloudEncoder {
    #[new]
    fn new(py: Python<'_>, info: Py<PyEncodingInfo>) -> PyResult<Self> {
        let core_info = info.borrow(py).to_core(py)?;
        Ok(Self {
            info,
            core_info,
            header: OnceLock::new(),
        })
    }

    #[getter]
    fn info(&self, py: Python<'_>) -> Py<PyEncodingInfo> {
        self.info.clone_ref(py)
    }

    #[getter]
    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(
            py,
            self.header_bytes().map_err(PyRuntimeError::new_err)?,
        ))
    }

    #[pyo3(signature = (
        cloud_data,
        *,
        drop_invalid=false,
        sort_field=None,
        is_bigendian=false,
        return_metadata=false
    ))]
    fn encode(
        &self,
        py: Python<'_>,
        cloud_data: &Bound<'_, PyAny>,
        drop_invalid: bool,
        sort_field: Option<&str>,
        is_bigendian: bool,
        return_metadata: bool,
    ) -> PyResult<Py<PyAny>> {
        if drop_invalid || sort_field.is_some() {
            let sort_field = sort_field.map(str::to_owned);
            let (_, result) = detach_with_input(py, cloud_data, |data| {
                if is_bigendian {
                    let normalized = big_endian_to_little(&self.core_info, data)?;
                    preprocess::encode(
                        &self.core_info,
                        &normalized,
                        drop_invalid,
                        sort_field.as_deref(),
                    )
                } else {
                    preprocess::encode(&self.core_info, data, drop_invalid, sort_field.as_deref())
                }
            })?;
            return encode_result_to_python(
                py,
                PyBytes::new(py, &result.data).unbind(),
                return_metadata,
                result.transformed_point_count,
                result.did_filter_invalid_xyz,
            );
        }

        if !is_bigendian
            && self.core_info.fields.len() > 1
            && let Some(encoded) = encode_borrowed_input(self, py, cloud_data)
        {
            return encode_result_to_python(py, encoded?.unbind(), return_metadata, None, false);
        }

        let mut data = buffer_to_vec(py, cloud_data)?;
        if is_bigendian {
            normalize_big_endian_in_place(&self.core_info, &mut data)
                .map_err(PyValueError::new_err)?;
        }
        let info = &self.core_info;
        let header = self.header_bytes().map_err(PyRuntimeError::new_err)?;
        let encoded = py
            .detach(|| codec::encode(info, header, &data))
            .map_err(PyRuntimeError::new_err)?;
        encode_result_to_python(
            py,
            PyBytes::new(py, &encoded).unbind(),
            return_metadata,
            None,
            false,
        )
    }

    #[pyo3(signature = (
        cloud_data,
        *,
        drop_invalid=true,
        sort_field=Some("line"),
        is_bigendian=false
    ))]
    fn preprocess(
        &self,
        py: Python<'_>,
        cloud_data: &Bound<'_, PyAny>,
        drop_invalid: bool,
        sort_field: Option<&str>,
        is_bigendian: bool,
    ) -> PyResult<(Py<PyBytes>, Option<u32>, bool)> {
        let sort_field = sort_field.map(str::to_owned);
        if is_bigendian {
            let data = buffer_to_vec(py, cloud_data)?;
            let info = &self.core_info;
            let (prepared, result) = py
                .detach(|| {
                    let prepared = big_endian_to_little(info, &data)?;
                    let result = preprocess::preprocess(
                        info,
                        &prepared,
                        drop_invalid,
                        sort_field.as_deref(),
                    )?;
                    Ok::<_, String>((prepared, result))
                })
                .map_err(PyRuntimeError::new_err)?;
            return match result.output {
                preprocess::PreprocessOutput::Unchanged => {
                    let point_count = u32::try_from(prepared.len() / info.point_step as usize)
                        .map_err(|_| {
                            PyValueError::new_err("Preprocessed point count exceeds u32")
                        })?;
                    Ok((
                        PyBytes::new(py, &prepared).unbind(),
                        Some(point_count),
                        result.did_filter_invalid_xyz,
                    ))
                }
                preprocess::PreprocessOutput::Changed(prepared, output_count) => Ok((
                    PyBytes::new(py, &prepared).unbind(),
                    Some(output_count),
                    result.did_filter_invalid_xyz,
                )),
            };
        }
        let (source, result) = detach_with_input(py, cloud_data, |data| {
            preprocess::preprocess(&self.core_info, data, drop_invalid, sort_field.as_deref())
        })?;
        match result.output {
            preprocess::PreprocessOutput::Unchanged => Ok((
                source.into_pyobject(py)?.unbind(),
                None,
                result.did_filter_invalid_xyz,
            )),
            preprocess::PreprocessOutput::Changed(prepared, output_count) => Ok((
                PyBytes::new(py, &prepared).unbind(),
                Some(output_count),
                result.did_filter_invalid_xyz,
            )),
        }
    }
}

impl PyPointcloudEncoder {
    fn header_bytes(&self) -> Result<&[u8], String> {
        if self.header.get().is_none() {
            let header_encoding = if self.core_info.version == 2 {
                codec::HeaderEncoding::Binary
            } else {
                codec::HeaderEncoding::Yaml
            };
            let header = codec::encode_header(&self.core_info, header_encoding)?;
            let _ = self.header.set(header);
        }
        Ok(self
            .header
            .get()
            .expect("header is initialized before access"))
    }
}

fn encode_result_to_python(
    py: Python<'_>,
    encoded: Py<PyBytes>,
    return_metadata: bool,
    transformed_point_count: Option<u32>,
    did_filter_invalid_xyz: bool,
) -> PyResult<Py<PyAny>> {
    if return_metadata {
        Ok((encoded, transformed_point_count, did_filter_invalid_xyz)
            .into_pyobject(py)?
            .into_any()
            .unbind())
    } else {
        Ok(encoded.into_any())
    }
}

fn big_endian_to_little(info: &EncodingInfo, data: &[u8]) -> Result<Vec<u8>, String> {
    let mut normalized = data.to_vec();
    normalize_big_endian_in_place(info, &mut normalized)?;
    Ok(normalized)
}

fn normalize_big_endian_in_place(info: &EncodingInfo, data: &mut [u8]) -> Result<(), String> {
    let point_step = info.point_step as usize;
    if point_step == 0 {
        return Err("point_step must be greater than zero".to_string());
    }
    if !data.len().is_multiple_of(point_step) {
        return Err("Input cloud_data size is not a multiple of point_step".to_string());
    }
    for field in &info.fields {
        let offset = field.offset as usize;
        let size = field.field_type.size_of();
        if size == 0 || offset.checked_add(size).is_none_or(|end| end > point_step) {
            return Err(format!(
                "Field '{}' exceeds point_step {}",
                field.name, info.point_step
            ));
        }
        if size == 1 {
            continue;
        }
        for point in data.chunks_exact_mut(point_step) {
            point[offset..offset + size].reverse();
        }
    }
    Ok(())
}

fn detach_with_input<T, F>(
    py: Python<'_>,
    cloud_data: &Bound<'_, PyAny>,
    operation: F,
) -> PyResult<(PyBackedBytes, T)>
where
    F: Send + FnOnce(&[u8]) -> Result<T, String>,
    T: Send,
{
    let data = py_backed_bytes(py, cloud_data)?;
    let (data, result) = py.detach(move || {
        let result = operation(&data);
        (data, result)
    });
    Ok((data, result.map_err(PyRuntimeError::new_err)?))
}

#[inline(never)]
fn encode_borrowed_input<'py>(
    encoder: &PyPointcloudEncoder,
    py: Python<'py>,
    cloud_data: &Bound<'py, PyAny>,
) -> Option<PyResult<Bound<'py, PyBytes>>> {
    let bytes = cloud_data.cast::<PyBytes>().ok()?;
    let owner = bytes.clone().unbind();
    let pointer = bytes.as_bytes().as_ptr() as usize;
    let length = bytes.as_bytes().len();
    let info = &encoder.core_info;
    let header = match encoder.header_bytes() {
        Ok(header) => header,
        Err(error) => return Some(Err(PyRuntimeError::new_err(error))),
    };
    if let Some(output_length) =
        codec::direct_lossless_xyz_output_length(info, header.len(), length)
    {
        let (encoded, output_pointer) = match allocate_private_bytes(py, output_length, true) {
            Ok(output) => output,
            Err(error) => return Some(Err(error)),
        };
        let result = py
            .detach(|| {
                let _owner = &owner;
                // Python bytes are immutable and `owner` keeps their allocation alive.
                let data = unsafe { std::slice::from_raw_parts(pointer as *const u8, length) };
                // The output bytes is not observable until this encode succeeds.
                let output = unsafe {
                    std::slice::from_raw_parts_mut(output_pointer as *mut u8, output_length)
                };
                codec::encode_direct_lossless_xyz_into(info, header, data, output)
            })
            .map_err(PyRuntimeError::new_err);
        drop(owner);
        return Some(result.map(|()| encoded));
    }
    let encoded = py
        .detach(|| {
            let _owner = &owner;
            // Python bytes are immutable and `owner` keeps their allocation alive.
            let data = unsafe { std::slice::from_raw_parts(pointer as *const u8, length) };
            codec::encode(info, header, data)
        })
        .map_err(PyRuntimeError::new_err);
    drop(owner);
    Some(encoded.map(|encoded| PyBytes::new(py, &encoded)))
}

#[pyclass(name = "PointcloudDecoder", module = "pureini")]
#[derive(Default)]
struct PyPointcloudDecoder;

#[pymethods]
impl PyPointcloudDecoder {
    #[new]
    fn new() -> Self {
        Self
    }

    fn decode(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<(Py<PyBytes>, Py<PyEncodingInfo>)> {
        if let Ok(bytes) = data.cast::<PyBytes>() {
            let (info, header_size) =
                codec::decode_header(bytes.as_bytes()).map_err(PyRuntimeError::new_err)?;
            return decode_borrowed_bytes(py, bytes, info, header_size);
        }

        let data = buffer_to_vec(py, data)?;
        let (info, header_size) = codec::decode_header(&data).map_err(PyRuntimeError::new_err)?;
        decode_owned_bytes(py, data, info, header_size)
    }
}

fn decoded_output_length(info: &EncodingInfo) -> PyResult<usize> {
    (info.width as usize)
        .checked_mul(info.height as usize)
        .and_then(|points| points.checked_mul(info.point_step as usize))
        .ok_or_else(|| PyRuntimeError::new_err("Decoded CloudINI size overflows usize"))
}

fn fields_cover_point(info: &EncodingInfo) -> bool {
    let point_step = info.point_step as usize;
    let mut ranges = info
        .fields
        .iter()
        .map(|field| {
            let start = field.offset as usize;
            (start, start.saturating_add(field.field_type.size_of()))
        })
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|range| range.0);

    let mut covered = 0;
    for (start, end) in ranges {
        if start > covered {
            return false;
        }
        covered = covered.max(end);
        if covered >= point_step {
            return true;
        }
    }
    covered >= point_step
}

fn allocate_private_bytes<'py>(
    py: Python<'py>,
    length: usize,
    is_fully_written: bool,
) -> PyResult<(Bound<'py, PyBytes>, usize)> {
    // The object remains private until decoding completes, so dense outputs need no initial fill.
    unsafe {
        let object = ffi::PyBytes_FromStringAndSize(std::ptr::null(), length as isize);
        let bytes = Bound::from_owned_ptr_or_err(py, object)?.cast_into_unchecked::<PyBytes>();
        let pointer = ffi::PyBytes_AsString(bytes.as_ptr()).cast::<u8>();
        if !is_fully_written {
            pointer.write_bytes(0, length);
        }
        Ok((bytes, pointer as usize))
    }
}

#[inline(never)]
fn decode_owned_bytes(
    py: Python<'_>,
    data: Vec<u8>,
    info: EncodingInfo,
    header_size: usize,
) -> PyResult<(Py<PyBytes>, Py<PyEncodingInfo>)> {
    let output_length = decoded_output_length(&info)?;
    let (decoded, pointer) = allocate_private_bytes(py, output_length, fields_cover_point(&info))?;
    py.detach(|| {
        let payload = data
            .get(header_size..)
            .ok_or_else(|| "CloudINI header exceeds input size".to_string())?;
        // The Python bytes is not observable until this decode succeeds.
        let output = unsafe { std::slice::from_raw_parts_mut(pointer as *mut u8, output_length) };
        codec::decode_payload_into(&info, payload, output)
    })
    .map_err(PyRuntimeError::new_err)?;
    Ok((
        decoded.unbind(),
        Py::new(py, PyEncodingInfo::from_core(py, info)?)?,
    ))
}

#[inline(never)]
fn decode_borrowed_bytes(
    py: Python<'_>,
    data: &Bound<'_, PyBytes>,
    info: EncodingInfo,
    header_size: usize,
) -> PyResult<(Py<PyBytes>, Py<PyEncodingInfo>)> {
    let output_length = decoded_output_length(&info)?;
    let (decoded, output_pointer) =
        allocate_private_bytes(py, output_length, fields_cover_point(&info))?;
    let owner = data.clone().unbind();
    let input_pointer = data.as_bytes().as_ptr() as usize;
    let input_length = data.as_bytes().len();
    let decode_info = &info;
    py.detach(move || {
        // Python bytes are immutable and `owner` keeps the input allocation alive.
        let input = unsafe { std::slice::from_raw_parts(input_pointer as *const u8, input_length) };
        let payload = input
            .get(header_size..)
            .ok_or_else(|| "CloudINI header exceeds input size".to_string())?;
        // The output bytes is not observable until this decode succeeds.
        let output =
            unsafe { std::slice::from_raw_parts_mut(output_pointer as *mut u8, output_length) };
        let result = codec::decode_payload_into(decode_info, payload, output);
        drop(owner);
        result
    })
    .map_err(PyRuntimeError::new_err)?;
    Ok((
        decoded.unbind(),
        Py::new(py, PyEncodingInfo::from_core(py, info)?)?,
    ))
}

#[pyclass(name = "BufferView", module = "pureini._pureini")]
struct BufferView {
    view: Py<PyAny>,
    offset: usize,
    length: usize,
}

#[pymethods]
impl BufferView {
    #[new]
    fn new(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let view = py
            .import("builtins")?
            .getattr("memoryview")?
            .call1((data,))?;
        let length = view.len()?;
        Ok(Self {
            view: view.unbind(),
            offset: 0,
            length,
        })
    }

    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.remaining_view(py)
    }

    fn size(&self) -> usize {
        self.length - self.offset
    }

    fn empty(&self) -> bool {
        self.size() == 0
    }

    fn trim_front(&mut self, count: usize) -> PyResult<()> {
        if count > self.size() {
            return Err(PyRuntimeError::new_err(format!(
                "Cannot trim {count} bytes, only {} available",
                self.size()
            )));
        }
        self.offset += count;
        Ok(())
    }

    fn write_bytes(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = buffer_to_vec(py, data)?;
        if bytes.len() > self.size() {
            return Err(PyRuntimeError::new_err(format!(
                "Cannot write {} bytes, only {} available",
                bytes.len(),
                self.size()
            )));
        }
        let slice = PySlice::new(
            py,
            self.offset as isize,
            (self.offset + bytes.len()) as isize,
            1,
        );
        self.view
            .bind(py)
            .set_item(slice, PyBytes::new(py, &bytes))?;
        self.offset += bytes.len();
        Ok(())
    }

    fn read_bytes<'py>(&mut self, py: Python<'py>, count: usize) -> PyResult<Bound<'py, PyBytes>> {
        if count > self.size() {
            return Err(PyRuntimeError::new_err(format!(
                "Cannot read {count} bytes, only {} available",
                self.size()
            )));
        }
        let slice = PySlice::new(py, self.offset as isize, (self.offset + count) as isize, 1);
        let selected = self.view.bind(py).get_item(slice)?;
        let bytes = buffer_to_vec(py, &selected)?;
        self.offset += count;
        Ok(PyBytes::new(py, &bytes))
    }
}

impl BufferView {
    fn remaining_view(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let slice = PySlice::new(py, self.offset as isize, self.length as isize, 1);
        Ok(self.view.bind(py).get_item(slice)?.unbind())
    }

    fn write_raw(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<()> {
        self.write_bytes(py, PyBytes::new(py, data).as_any())
    }

    fn read_raw(&mut self, py: Python<'_>, count: usize) -> PyResult<Vec<u8>> {
        Ok(self.read_bytes(py, count)?.as_bytes().to_vec())
    }
}

#[pyfunction]
#[pyo3(signature = (value, buffer, offset=0))]
fn encode_varint64_to_buffer(
    py: Python<'_>,
    value: i64,
    buffer: &Bound<'_, PyAny>,
    offset: usize,
) -> PyResult<usize> {
    if value == i64::MIN {
        return Err(PyValueError::new_err(
            "CloudINI varint cannot represent -9223372036854775808",
        ));
    }
    let mut encoded = [0; 10];
    let size = varint::encode(value, &mut encoded);
    let view = py
        .import("builtins")?
        .getattr("memoryview")?
        .call1((buffer,))?;
    let slice = PySlice::new(py, offset as isize, (offset + size) as isize, 1);
    view.set_item(slice, PyBytes::new(py, &encoded[..size]))?;
    Ok(size)
}

#[pyfunction]
#[pyo3(signature = (data, offset=0))]
fn decode_varint(py: Python<'_>, data: &Bound<'_, PyAny>, offset: usize) -> PyResult<(i64, usize)> {
    let bytes = buffer_to_vec(py, data)?;
    varint::decode(
        bytes
            .get(offset..)
            .ok_or_else(|| PyRuntimeError::new_err("Incomplete varint in buffer"))?,
    )
    .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
fn encode(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    mut buffer: PyRefMut<'_, BufferView>,
    format_char: &str,
) -> PyResult<()> {
    let bytes = encode_primitive(value, format_char)?;
    buffer.write_raw(py, &bytes)
}

#[pyfunction]
fn decode(
    py: Python<'_>,
    mut buffer: PyRefMut<'_, BufferView>,
    format_char: &str,
) -> PyResult<Py<PyAny>> {
    let size = primitive_size(format_char)?;
    let bytes = buffer.read_raw(py, size)?;
    decode_primitive(py, &bytes, format_char)
}

#[pyfunction]
fn encode_string(
    py: Python<'_>,
    value: &str,
    mut buffer: PyRefMut<'_, BufferView>,
) -> PyResult<()> {
    let bytes = value.as_bytes();
    let length = u16::try_from(bytes.len()).map_err(|_| {
        PyValueError::new_err(format!(
            "String too long: {} bytes (max 65535)",
            bytes.len()
        ))
    })?;
    buffer.write_raw(py, &length.to_le_bytes())?;
    buffer.write_raw(py, bytes)
}

#[pyfunction]
fn decode_string(py: Python<'_>, mut buffer: PyRefMut<'_, BufferView>) -> PyResult<String> {
    let length = u16::from_le_bytes(buffer.read_raw(py, 2)?.try_into().unwrap()) as usize;
    String::from_utf8(buffer.read_raw(py, length)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn build_field_metadata(
    py: Python<'_>,
    info: PyRef<'_, PyEncodingInfo>,
) -> PyResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
    let core = info.to_core(py)?;
    let offsets = core.fields.iter().map(|field| field.offset).collect();
    let field_types = core
        .fields
        .iter()
        .map(|field| field.field_type as u32)
        .collect();
    let resolutions = core
        .fields
        .iter()
        .map(|field| {
            if core.encoding_opt == EncodingOptions::Lossy
                && let Some(resolution) = field.resolution
            {
                return f64::from(resolution);
            }
            match field.field_type {
                FieldType::Float32 => -1.0,
                _ => 0.0,
            }
        })
        .collect();
    Ok((offsets, field_types, resolutions))
}

#[pyfunction]
fn encoding_info_to_yaml(py: Python<'_>, info: PyRef<'_, PyEncodingInfo>) -> PyResult<String> {
    Ok(codec::encoding_info_to_yaml(&info.to_core(py)?))
}

#[pyfunction]
fn encoding_info_from_yaml(py: Python<'_>, yaml: &str) -> PyResult<Py<PyEncodingInfo>> {
    let info = codec::encoding_info_from_yaml(yaml).map_err(PyRuntimeError::new_err)?;
    Py::new(py, PyEncodingInfo::from_core(py, info)?)
}

#[pyfunction]
#[pyo3(signature = (info, encoding=1))]
fn encode_header(
    py: Python<'_>,
    info: PyRef<'_, PyEncodingInfo>,
    encoding: u8,
) -> PyResult<Py<PyBytes>> {
    let encoding = match encoding {
        0 => codec::HeaderEncoding::Binary,
        1 => codec::HeaderEncoding::Yaml,
        value => {
            return Err(PyValueError::new_err(format!(
                "Invalid HeaderEncoding {value}"
            )));
        }
    };
    let header =
        codec::encode_header(&info.to_core(py)?, encoding).map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &header).unbind())
}

#[pyfunction]
fn decode_header(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<(Py<PyEncodingInfo>, usize)> {
    let data = buffer_to_vec(py, data)?;
    let (info, size) = codec::decode_header(&data).map_err(PyRuntimeError::new_err)?;
    Ok((Py::new(py, PyEncodingInfo::from_core(py, info)?)?, size))
}

#[pyfunction]
fn compute_header_size(py: Python<'_>, fields: &Bound<'_, PyAny>) -> PyResult<usize> {
    let mut size = 10 + 2 + 4 + 4 + 4 + 1 + 1 + 2;
    for field in fields.try_iter()? {
        let field = field?.extract::<PyRef<'_, PyPointField>>()?;
        size += 2 + field.name.len() + 4 + 1 + 4;
    }
    let _ = py;
    Ok(size)
}

#[pymodule]
#[pyo3(name = "_pureini")]
fn pureini(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__implementation__", "rust")?;
    module.add("ENCODING_VERSION", codec::ENCODING_VERSION)?;
    module.add("MAGIC_HEADER", PyBytes::new(py, b"CLOUDINI_V"))?;
    module.add("MAGIC_HEADER_LENGTH", 10)?;
    module.add("DECODE_BUT_SKIP_STORE", u32::MAX)?;
    module.add("POINTS_PER_CHUNK", POINTS_PER_CHUNK)?;

    let field_type_class = create_int_enum(
        py,
        "FieldType",
        &[
            ("UNKNOWN", 0),
            ("INT8", 1),
            ("UINT8", 2),
            ("INT16", 3),
            ("UINT16", 4),
            ("INT32", 5),
            ("UINT32", 6),
            ("FLOAT32", 7),
            ("FLOAT64", 8),
            ("INT64", 9),
            ("UINT64", 10),
        ],
    )?;
    let encoding_options_class = create_int_enum(
        py,
        "EncodingOptions",
        &[("NONE", 0), ("LOSSY", 1), ("LOSSLESS", 2)],
    )?;
    let compression_option_class = create_int_enum(
        py,
        "CompressionOption",
        &[("NONE", 0), ("LZ4", 1), ("ZSTD", 2)],
    )?;
    let header_encoding_class =
        create_int_enum(py, "HeaderEncoding", &[("BINARY", 0), ("YAML", 1)])?;
    module.add("FieldType", field_type_class)?;
    module.add("EncodingOptions", encoding_options_class)?;
    module.add("CompressionOption", compression_option_class)?;
    module.add("HeaderEncoding", header_encoding_class)?;

    module.add_class::<PyPointField>()?;
    module.add_class::<PyEncodingInfo>()?;
    module.add_class::<PyPointcloudEncoder>()?;
    module.add_class::<PyPointcloudDecoder>()?;
    module.add_class::<BufferView>()?;
    module.add_function(wrap_pyfunction!(encode_varint64_to_buffer, module)?)?;
    module.add_function(wrap_pyfunction!(decode_varint, module)?)?;
    module.add_function(wrap_pyfunction!(encode, module)?)?;
    module.add_function(wrap_pyfunction!(decode, module)?)?;
    module.add_function(wrap_pyfunction!(encode_string, module)?)?;
    module.add_function(wrap_pyfunction!(decode_string, module)?)?;
    module.add_function(wrap_pyfunction!(build_field_metadata, module)?)?;
    module.add_function(wrap_pyfunction!(encoding_info_to_yaml, module)?)?;
    module.add_function(wrap_pyfunction!(encoding_info_from_yaml, module)?)?;
    module.add_function(wrap_pyfunction!(encode_header, module)?)?;
    module.add_function(wrap_pyfunction!(decode_header, module)?)?;
    module.add_function(wrap_pyfunction!(compute_header_size, module)?)?;

    module.add(
        "__all__",
        PyList::new(
            py,
            [
                "CompressionOption",
                "EncodingInfo",
                "EncodingOptions",
                "FieldType",
                "PointField",
                "PointcloudEncoder",
                "PointcloudDecoder",
            ],
        )?,
    )?;
    Ok(())
}

fn create_int_enum(py: Python<'_>, name: &str, members: &[(&str, u8)]) -> PyResult<Py<PyAny>> {
    let values = PyDict::new(py);
    for (member_name, value) in members {
        values.set_item(member_name, value)?;
    }
    let class = py
        .import("enum")?
        .getattr("IntEnum")?
        .call1((name, values))?;
    class.setattr("__module__", "pureini")?;
    Ok(class.unbind())
}

fn enum_member(py: Python<'_>, enum_name: &str, value: u8) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("pureini")?
        .getattr(enum_name)?
        .call1((value,))?
        .unbind())
}

fn field_type(value: u8) -> PyResult<FieldType> {
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
        _ => Err(PyValueError::new_err(format!("Invalid FieldType {value}"))),
    }
}

fn encoding_option(value: u8) -> PyResult<EncodingOptions> {
    match value {
        0 => Ok(EncodingOptions::None),
        1 => Ok(EncodingOptions::Lossy),
        2 => Ok(EncodingOptions::Lossless),
        _ => Err(PyValueError::new_err(format!(
            "Invalid EncodingOptions {value}"
        ))),
    }
}

fn compression_option(value: u8) -> PyResult<CompressionOption> {
    match value {
        0 => Ok(CompressionOption::None),
        1 => Ok(CompressionOption::Lz4),
        2 => Ok(CompressionOption::Zstd),
        _ => Err(PyValueError::new_err(format!(
            "Invalid CompressionOption {value}"
        ))),
    }
}

fn buffer_to_vec(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Ok(bytes.as_bytes().to_vec());
    }
    let bytes = py.import("builtins")?.getattr("bytes")?.call1((value,))?;
    Ok(bytes.cast::<PyBytes>()?.as_bytes().to_vec())
}

fn py_backed_bytes(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyBackedBytes> {
    if let Ok(data) = value.extract::<PyBackedBytes>() {
        return Ok(data);
    }
    let bytes = py
        .import("builtins")?
        .getattr("bytes")?
        .call1((value,))?
        .cast_into::<PyBytes>()?;
    Ok(PyBackedBytes::from(bytes))
}

fn iterable_to_list(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyList>> {
    let list = PyList::empty(py);
    for item in value.try_iter()? {
        list.append(item?)?;
    }
    Ok(list.unbind())
}

fn primitive_size(format_char: &str) -> PyResult<usize> {
    match format_char {
        "b" | "B" => Ok(1),
        "h" | "H" => Ok(2),
        "i" | "I" | "f" => Ok(4),
        "q" | "Q" | "d" => Ok(8),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported struct format '{format_char}'"
        ))),
    }
}

fn encode_primitive(value: &Bound<'_, PyAny>, format_char: &str) -> PyResult<Vec<u8>> {
    let bytes = match format_char {
        "b" => value.extract::<i8>()?.to_le_bytes().to_vec(),
        "B" => value.extract::<u8>()?.to_le_bytes().to_vec(),
        "h" => value.extract::<i16>()?.to_le_bytes().to_vec(),
        "H" => value.extract::<u16>()?.to_le_bytes().to_vec(),
        "i" => value.extract::<i32>()?.to_le_bytes().to_vec(),
        "I" => value.extract::<u32>()?.to_le_bytes().to_vec(),
        "q" => value.extract::<i64>()?.to_le_bytes().to_vec(),
        "Q" => value.extract::<u64>()?.to_le_bytes().to_vec(),
        "f" => value.extract::<f32>()?.to_le_bytes().to_vec(),
        "d" => value.extract::<f64>()?.to_le_bytes().to_vec(),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported struct format '{format_char}'"
            )));
        }
    };
    Ok(bytes)
}

fn decode_primitive(py: Python<'_>, bytes: &[u8], format_char: &str) -> PyResult<Py<PyAny>> {
    let value = match format_char {
        "b" => i8::from_le_bytes(bytes.try_into().unwrap())
            .into_pyobject(py)?
            .into_any(),
        "B" => u8::from_le_bytes(bytes.try_into().unwrap())
            .into_pyobject(py)?
            .into_any(),
        "h" => i16::from_le_bytes(bytes.try_into().unwrap())
            .into_pyobject(py)?
            .into_any(),
        "H" => u16::from_le_bytes(bytes.try_into().unwrap())
            .into_pyobject(py)?
            .into_any(),
        "i" => i32::from_le_bytes(bytes.try_into().unwrap())
            .into_pyobject(py)?
            .into_any(),
        "I" => u32::from_le_bytes(bytes.try_into().unwrap())
            .into_pyobject(py)?
            .into_any(),
        "q" => i64::from_le_bytes(bytes.try_into().unwrap())
            .into_pyobject(py)?
            .into_any(),
        "Q" => u64::from_le_bytes(bytes.try_into().unwrap())
            .into_pyobject(py)?
            .into_any(),
        "f" => f32::from_le_bytes(bytes.try_into().unwrap())
            .into_pyobject(py)?
            .into_any(),
        "d" => f64::from_le_bytes(bytes.try_into().unwrap())
            .into_pyobject(py)?
            .into_any(),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported struct format '{format_char}'"
            )));
        }
    };
    Ok(value.unbind())
}
