import ektools
import wavelets as W
import numpy as np

# Get a RAW3 datagram
INDEX = 108
ix = ektools.index('../CRIMAC-KoronaScript/cr_out/2023001004-D20230321-T082157.raw')
assert ix[INDEX][1] == 'RAW3', f'Wrong datagram? {ix[INDEX][:3]}'
dg = ektools.parse(ix[INDEX][3])

HPSEARCH = False
PRINT = False
PLOT = True
TIME = False
CLIP = True

# Using log data completely breaks everything, as most of the variance is now in the very low values.
# (e.g. noise that varies between 1e-9 and 1e-8 now becomes -8 to -9, a differenc of 1 same as the
# much more important difference between 1e0 and 1e-1, or 1 and 0.1)

# mydata = np.log(dg['complex'][:, 0].copy())
mydata = dg['complex'][:, 0].copy()
mydatasize = mydata.size * mydata.itemsize

if CLIP:
    mydata = np.clip(np.abs(mydata), 0, 1e-3) * np.exp(1j * np.angle(mydata))

print('Input shape:', mydata.shape, ' size:', mydatasize)


if HPSEARCH:
  for level in [1, 2, 3, 4, 5, 6, 8, 10]:
    # print('level:', level)
    for thresh in [0.005, 0.01, 0.025, 0.033, 0.05, 0.1, 0.15, 0.2]:
        # print('thresh:', thresh)
        compressed, wvl, lev, shp = W.compress(mydata, level=level, threshold_ratio=thresh)
        compsize = len(compressed[0]) + len(compressed[1])
        # print(compsize)

        reconstructed = W.decompress(compressed, wvl, lev, shp)[:mydata.shape[0]]
        if PRINT:
            print(mydata[PRINT:PRINT + 10])
            print(reconstructed[PRINT:PRINT + 10])

        err = np.abs(mydata - reconstructed)
        mse = np.mean(err ** 2)
        relerr = np.mean(err / np.abs(mydata))
        print(f'level={level:<2d} thresh={thresh:.3f} size={compsize / 1000:3.1f}k reduction={(1 - compsize / mydatasize) * 100:.1f}% mse={mse:.3e} rel={relerr:.2f}')
    print()


import numpy as np
import matplotlib.pyplot as plt

if PLOT:
    # Plot it
    r1, r2, r3 = 0.20, 0.10, 0.05
    compressed, wvl, lev, shp = W.compress(mydata, level=3, threshold_ratio=r1)
    reconstr1 = W.decompress(compressed, wvl, lev, shp)[:mydata.shape[0]]
    compressed, wvl, lev, shp = W.compress(mydata, level=2, threshold_ratio=r2)
    reconstr2 = W.decompress(compressed, wvl, lev, shp)[:mydata.shape[0]]
    compressed, wvl, lev, shp = W.compress(mydata, level=2, threshold_ratio=r3)
    reconstr3 = W.decompress(compressed, wvl, lev, shp)[:mydata.shape[0]]


    # Assume mydata and reconstructed are your complex signals
    plt.figure(figsize=(10, 5))
    plt.plot(np.abs(mydata), label='Original', alpha=0.7)
    # plt.plot(np.abs(reconstr1), label=f'Level=3 thr={r1}', alpha=0.7)
    plt.plot(np.abs(reconstr2), label=f'Level=2 thr={r2}', alpha=0.7)
    plt.plot(np.abs(mydata-reconstr2), label='Error', alpha=0.7)
    # plt.plot(np.abs(reconstr2), label=f'Level=2 thr={r3}', alpha=0.7)
    plt.title('Signal Magnitude: Original vs Reconstructed')
    plt.xlabel('Sample Index')
    plt.ylabel('Magnitude')
    #plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()


if TIME:
    # Time it.
    import timeit

    compressed, wvl, lev, shp = W.compress(mydata, level=3, threshold_ratio=0.025)
    compress_time = timeit.timeit(lambda: W.compress(mydata, level=3, threshold_ratio=0.025), number=100) / 100
    decompress_time = timeit.timeit(lambda: W.decompress(compressed, wvl, lev, shp), number=100) / 100
    print(f"Average compression time: {compress_time:.4f} seconds")
    print(f"Average decompression time: {decompress_time:.4f} seconds")
