# codec.py
import sys
import struct
import zlib
import numpy as np
from scipy.io import wavfile


# ============================================================
# 変更するのはここ(predict)だけ。
# x[:n] は既知の過去信号値
# 現在値(x[n])や未来の値は使用してはいけない(使用しても可逆チェックで落ちる)
# 必ず整数を返すこと!
# ============================================================
def predict(x, n):
    if n == 0:
        return 0
    return int(x[n-1])
# ============================================================


def zigzag(e):
    """signed integer -> unsigned integer"""
    e = np.asarray(e, dtype=np.int64)
    u = np.where(e >= 0, 2 * e, -2 * e - 1)

    if np.any(u > np.iinfo(np.uint32).max):
        raise ValueError("Prediction error is too large.")

    return u.astype(np.uint32)


def unzigzag(u):
    """unsigned integer -> signed integer"""
    u = np.asarray(u, dtype=np.uint64)
    return np.where(
        u & 1,
        -((u >> 1).astype(np.int64)) - 1,4$
        (u >> 1).astype(np.int64)
    )


def decode_data(data, N):
    """compressed data -> PCM samples"""
    u = np.frombuffer(zlib.decompress(data), dtype=np.uint32)

    if len(u) != N:
        raise ValueError("Invalid compressed data.")

    e = unzigzag(u)
    x = np.empty(N, dtype=np.int64)

    for n in range(N):
        x[n] = predict(x, n) + e[n]

    if np.any(x < -32768) or np.any(x > 32767):
        raise ValueError("Decoded samples are outside int16 range.")

    return x.astype(np.int16)


def encode(infile, outfile):
    fs, x = wavfile.read(infile)

    if x.ndim != 1 or x.dtype != np.int16:
        raise ValueError("Input must be mono 16-bit PCM WAV.")

    # prediction error
    xx = x.astype(np.int64)
    e = np.empty(len(xx), dtype=np.int64)

    for n in range(len(xx)):
        e[n] = xx[n] - predict(xx, n)

    # ZigZag + DEFLATE
    u = zigzag(e)
    data = zlib.compress(u.tobytes(), level=9)

    # header: sampling frequency, number of samples
    header = struct.pack("<II", fs, len(x)) # "<II": little endian, uint32x2

    with open(outfile, "wb") as f:
        f.write(header)
        f.write(data)

    # automatic lossless check
    y = decode_data(data, len(x))

    if not np.array_equal(x, y):
        raise RuntimeError("Lossless check: NG")

    print("Lossless check : OK")
    print(f"Samples        : {len(x)}")
    print(f"Original PCM   : {x.nbytes} bytes")
    print(f"Compressed     : {len(header) + len(data)} bytes")
    print(f"Ratio          : {(len(header) + len(data)) / x.nbytes:.4f}")


def decode(infile, outfile):
    with open(infile, "rb") as f:
        fs, N = struct.unpack("<II", f.read(8)) # "<II": little endian, uint32x2
        data = f.read()

    x = decode_data(data, N)
    wavfile.write(outfile, fs, x)

    print(f"Decoded        : {outfile}")
    print(f"Samples        : {N}")
    print(f"Sampling rate  : {fs} Hz")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("e", "d"):
        print("Encode: python codec.py e input.wav output.bin")
        print("Decode: python codec.py d input.bin output.wav")
        sys.exit(1)

    if sys.argv[1] == "e":
        encode(sys.argv[2], sys.argv[3])
    else:
        decode(sys.argv[2], sys.argv[3])