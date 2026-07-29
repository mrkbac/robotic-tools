"""Decode fixtures produced by the CloudINI C++ reference."""

import base64
import struct

import pytest
from pureini import PointcloudDecoder, PointcloudEncoder

_CPP_ZSTD_FIXTURES = {
    3: (
        "Q0xPVURJTklfVjAzCnZlcnNpb246IDMKd2lkdGg6IDQwOTcKaGVpZ2h0OiAxCnBvaW50X3N0ZXA6IDQw"
        "CmVuY29kaW5nX29wdDogTE9TU1kKY29tcHJlc3Npb25fb3B0OiBaU1RECmZpZWxkczoKICAtIG5hbWU6"
        "IHgKICAgIG9mZnNldDogMAogICAgdHlwZTogRkxPQVQzMgogICAgcmVzb2x1dGlvbjogMC4wMDEKICAt"
        "IG5hbWU6IHkKICAgIG9mZnNldDogNAogICAgdHlwZTogRkxPQVQzMgogICAgcmVzb2x1dGlvbjogMC4w"
        "MDEKICAtIG5hbWU6IHoKICAgIG9mZnNldDogOAogICAgdHlwZTogRkxPQVQzMgogICAgcmVzb2x1dGlv"
        "bjogMC4wMDEKICAtIG5hbWU6IGkxNgogICAgb2Zmc2V0OiAxMgogICAgdHlwZTogSU5UMTYKICAgIHJl"
        "c29sdXRpb246IG51bGwKICAtIG5hbWU6IHUxNgogICAgb2Zmc2V0OiAxNAogICAgdHlwZTogVUlOVDE2"
        "CiAgICByZXNvbHV0aW9uOiBudWxsCiAgLSBuYW1lOiBpMzIKICAgIG9mZnNldDogMTYKICAgIHR5cGU6"
        "IElOVDMyCiAgICByZXNvbHV0aW9uOiBudWxsCiAgLSBuYW1lOiB1MzIKICAgIG9mZnNldDogMjAKICAg"
        "IHR5cGU6IFVJTlQzMgogICAgcmVzb2x1dGlvbjogbnVsbAogIC0gbmFtZTogaTY0CiAgICBvZmZzZXQ6"
        "IDI0CiAgICB0eXBlOiBJTlQ2NAogICAgcmVzb2x1dGlvbjogbnVsbAogIC0gbmFtZTogdTY0CiAgICBv"
        "ZmZzZXQ6IDMyCiAgICB0eXBlOiBVSU5UNjQKICAgIHJlc29sdXRpb246IG51bGwKAEgDAAAotS/9YAqP"
        "9RkACoXABhDApdAB////P/D/Hzi/lrRVaQBlAGUAbdu2bdu2bdu2bctt27Zt27Zt27Zt27bctm3btm3b"
        "tm3btm3Lbdu2bdu2bdu2bdu23LZt27Zt27Zt27Zty23btm3btm3btm3btty2bdu2bdu2bdu2bctt27Zt"
        "27Zt27Zt27aIkJNARpAEbdu2bdu2bdu2bdty27Zt27Zt27Zt27Ytt23btm3btm3btm3bctu2bdu2bdu2"
        "bdu2Lbdt27Zt27Zt27Zt23Lbtm3btm3btm3bti23bdu2bdu2bdu2bdty27Zt27Zt27Zt27Ytt21t27Zt"
        "27Zt27Zt27bctm3btm3btm3btm3Lbdu2bdu2bdu2bdu23LZt27Zt27Zt27Zty23btm3btm3btm3btty2"
        "bdu2bdu2bdu2bctt27Zt27Zt27Zt27bctm3btm3btm3btm3LbRFh27Zt27Zt27Zt27Ytt23btm3btm3b"
        "tm3bctu2bdu2bdu2bdu2Lbdt27Zt27Zt27Zt23Lbtm3btm3btm3bti23bdu2bdu2bdu2bdty27Zt27Zt"
        "27Zt27Ytt23btm3btm3btm3bcogAqBMAfpfN478GE4BAQAV8gX8EgQp/+efff/7/5/8f/eh0fv/n/3/"
        "+/efn6aL6f/7/53v//G+OcTq///P/P//+8/N0Uf0////z/T//m2Oczu//7P+ff//5ebqo/p////n+n//"
        "NMU7n93/+/+fff36eLqr3z///fP/P/+YYp/P7P///8+8/P08X1f/z/z/f/+N/c4zT+f2f///595+fp4vq"
        "//n/n+//+d8c43R+/+f/9/z7z8/TRfX//P/P9//8b45xOr//8/8///7z83RR/T/+/+f7f/43xzid3//5"
        "/59///l5uqj+n///+f6f/80xTs7v//z/z7///DxdVP/P//98/8//5hin8/s////z3X9+ni6q/+f/f77/"
        "539zjNP5/Z////n3n5+ni+r/+X8/3//zvznG6fz+z////PvPz9NF9f/8/8/3//xvjnE6X//n/3/+/efn"
        "6aL6f/7/5/t//jfHOJ3f//n/n39/8/N0Uf0////z/T//m2Oczu///P/Pv/+o/vIRsAvgSzc="
    ),
    5: (
        "Q0xPVURJTklfVjA1CnZlcnNpb246IDUKd2lkdGg6IDQwOTcKaGVpZ2h0OiAxCnBvaW50X3N0ZXA6IDQw"
        "CmVuY29kaW5nX29wdDogTE9TU1kKY29tcHJlc3Npb25fb3B0OiBaU1RECmZpZWxkczoKICAtIG5hbWU6"
        "IHgKICAgIG9mZnNldDogMAogICAgdHlwZTogRkxPQVQzMgogICAgcmVzb2x1dGlvbjogMC4wMDEKICAt"
        "IG5hbWU6IHkKICAgIG9mZnNldDogNAogICAgdHlwZTogRkxPQVQzMgogICAgcmVzb2x1dGlvbjogMC4w"
        "MDEKICAtIG5hbWU6IHoKICAgIG9mZnNldDogOAogICAgdHlwZTogRkxPQVQzMgogICAgcmVzb2x1dGlv"
        "bjogMC4wMDEKICAtIG5hbWU6IGkxNgogICAgb2Zmc2V0OiAxMgogICAgdHlwZTogSU5UMTYKICAgIHJl"
        "c29sdXRpb246IG51bGwKICAtIG5hbWU6IHUxNgogICAgb2Zmc2V0OiAxNAogICAgdHlwZTogVUlOVDE2"
        "CiAgICByZXNvbHV0aW9uOiBudWxsCiAgLSBuYW1lOiBpMzIKICAgIG9mZnNldDogMTYKICAgIHR5cGU6"
        "IElOVDMyCiAgICByZXNvbHV0aW9uOiBudWxsCiAgLSBuYW1lOiB1MzIKICAgIG9mZnNldDogMjAKICAg"
        "IHR5cGU6IFVJTlQzMgogICAgcmVzb2x1dGlvbjogbnVsbAogIC0gbmFtZTogaTY0CiAgICBvZmZzZXQ6"
        "IDI0CiAgICB0eXBlOiBJTlQ2NAogICAgcmVzb2x1dGlvbjogbnVsbAogIC0gbmFtZTogdTY0CiAgICBv"
        "ZmZzZXQ6IDMyCiAgICB0eXBlOiBVSU5UNjQKICAgIHJlc29sdXRpb246IG51bGwKAFIAAAAotS/9YL4z"
        "RQIAJAMBAdEPAwEBAgAAAAEBA4AgAQQAAAABAAIAAwDkAANAAAAAAYABAwEBfweAIApVAQGAIAYAIIuW"
        "PJ/kDLi79OePq9j+k3dG"
    ),
}

# Generated with CloudINI reference commit d202e5255d12519ff1f3db1dac4df3ca0e550ee7.
# FLOAT64 without a resolution exercises the V4 Gorilla lossless codec.
_CPP_V4_FLOAT64_FIXTURE = (
    "Q0xPVURJTklfVjA0CnZlcnNpb246IDQKd2lkdGg6IDEyCmhlaWdodDogMQpwb2ludF9zdGVwOiA4"
    "CmVuY29kaW5nX29wdDogTE9TU0xFU1MKY29tcHJlc3Npb25fb3B0OiBOT05FCmZpZWxkczoKICAt"
    "IG5hbWU6IHZhbHVlCiAgICBvZmZzZXQ6IDAKICAgIHR5cGU6IEZMT0FUNjQKICAgIHJlc29sdXRp"
    "b246IG51bGwKADoAAAAAAAAAAADwPwAzIAB/MAAAAAAHPwAAAAAAAP8PA6cACAAVgAEBAAGDv2+Y"
    "sqQ1SZYaNRlrBA/aZv4B"
)
_CPP_V4_FLOAT64_VALUES = (
    1.0,
    1.0,
    1.5,
    1.5,
    1.5000000000000002,
    2.0,
    -3.25,
    -3.25,
    0.0,
    -0.0,
    1e100,
    1e-100,
)

# Generated with the CloudINI C++ 0.5.0 reference, commit
# 8a5da14ae15367cc09e0ac8c5a66afb44f390110.
_CPP_V2_FIXTURE = (
    "Q0xPVURJTklfVjAyDAAAAAEAAAAOAAAAAQAEAAEAeAAAAAAHbxKDOgEAeQQAAAAHbxKDOgEA"
    "eggAAAAHbxKDOgQAcmluZwwAAAAEAACAvwEB0Q8BAwQDAwMEAwMDBAMDAwQDBgMEAwMDBAMD"
    "AwQDAwMEAwYDBAMDAwQDAwMEAwM="
)


def _expected_point(index: int) -> tuple[float, float, float, int, int, int, int, int, int]:
    return (
        index * 0.001,
        0.0,
        1.0,
        index % 10_000,
        index % 4,
        index // 128,
        index * 3,
        -index * 5,
        42,
    )


def _v2_source() -> bytes:
    return b"".join(
        struct.pack("<fffH", index * 0.001, -index * 0.002, 1.0 + index * 0.001, index % 4)
        for index in range(12)
    )


def test_decode_cpp_v2_reference_fixture():
    decoded, info = PointcloudDecoder().decode(base64.b64decode(_CPP_V2_FIXTURE))

    assert info.version == 2
    for index, point in enumerate(struct.iter_unpack("<fffH", decoded)):
        expected = struct.unpack_from("<fffH", _v2_source(), index * 14)
        assert point[:3] == pytest.approx(expected[:3], abs=0.00051)
        assert point[3] == expected[3]


def test_encode_matches_cpp_v2_reference_fixture():
    reference = base64.b64decode(_CPP_V2_FIXTURE)
    _, info = PointcloudDecoder().decode(reference)

    assert PointcloudEncoder(info).encode(_v2_source()) == reference


@pytest.mark.parametrize(("version", "fixture"), _CPP_ZSTD_FIXTURES.items())
def test_decode_cpp_reference_zstd_fixture(version: int, fixture: str):
    decoded, info = PointcloudDecoder().decode(base64.b64decode(fixture))

    assert info.version == version
    assert info.width == 4097
    for index, point in enumerate(struct.iter_unpack("<fffhHiIqQ", decoded)):
        expected = _expected_point(index)
        assert point[:3] == pytest.approx(expected[:3], abs=0.00051)
        assert point[3:] == expected[3:]


@pytest.mark.parametrize(("version", "fixture"), _CPP_ZSTD_FIXTURES.items())
def test_encode_matches_cpp_reference_zstd_fixture(version: int, fixture: str):
    reference = base64.b64decode(fixture)
    _, info = PointcloudDecoder().decode(reference)
    source = b"".join(
        struct.pack("<fffhHiIqQ", *_expected_point(index)) for index in range(info.width)
    )

    assert info.version == version
    assert PointcloudEncoder(info).encode(source) == reference


def test_decode_cpp_reference_v4_float64_fixture():
    decoded, info = PointcloudDecoder().decode(base64.b64decode(_CPP_V4_FLOAT64_FIXTURE))

    assert info.version == 4
    assert info.width == len(_CPP_V4_FLOAT64_VALUES)
    assert decoded == struct.pack("<12d", *_CPP_V4_FLOAT64_VALUES)


def test_encode_matches_cpp_reference_v4_float64_fixture():
    reference = base64.b64decode(_CPP_V4_FLOAT64_FIXTURE)
    source, info = PointcloudDecoder().decode(reference)

    assert PointcloudEncoder(info).encode(source) == reference
