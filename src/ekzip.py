# import numpy as np
import sys
import struct
from ektools import ekfile  # , parse
from ektools.simrad_parsers import SimradRawParser
from simrad_compressed_parser import SimradRawZParser
import numpy as np
import wavelets as W
import argparse
import os


# or just use dgram_write from eksplit?  Or finalize_datagram?
def dgram_write(f, dgram):
    """Write a datagram to a file"""
    hdr = struct.pack('<l', len(dgram))
    f.write(hdr)
    f.write(dgram)
    f.write(hdr)


def raw2raz(data, wavelet='db4', level=3, threshold_ratio=0.2):  # (dgram):
    '''Convert the dict representing a RAW type datagram into a RAZ compressed datatgram'''
    data = data.copy()
    match data['type']:
        case 'RAW3':
            data['type'] = 'RAZ' + data['type'][3]
            data['zlevel'] = level

            if data['n_complex'] > 0:
                zcomplex = []
                for i in range(data['n_complex']):
                    zd, wl, lv, sh = W.compress(data['complex'][:, i],
                                                wavelet=wavelet, level=level, threshold_ratio=threshold_ratio)
                    zcomplex.append(zd)  # oh fuck, it's a tuple, real/imag

                data['zlevel'] = lv
                data['zshapes'] = [s[0] for s in sh]
                data['zcomplex'] = zcomplex
                del data['complex']
            if data['power'] is not None:  # set to None if not present by the Simrad parser
                zd, wl, lv, sh = W.compress1(data['power'], wavelet=wavelet, level=level, threshold_ratio=threshold_ratio)
                data['zpower'] = zd
                data['zpshapes'] = [s[0] for s in sh]
                del data['power']
            if data['angle'] is not None:  # as above
                for i in range(data['count']):
                    pass
        case _:
            assert False, f'Datagram type {data['type']} not supported.'

    return data


def raz2raw(data):  # dgram:
    '''Convert the dict representing a RAZ compressed datagram into a RAW datatgram'''
    data = data.copy()
    match data['type']:
        case 'RAZ3':
            data['type'] = 'RAW' + data['type'][3]
            level = data['zlevel']

            if data['n_complex'] > 0:
                shapes = [(s,) for s in data['zshapes']]
                complex = []
                for i in range(data['n_complex']):
                    zd = W.decompress(data['zcomplex'][i], 'db4', level=level, shapes=shapes)
                    complex.append(zd[:data['count']])
                data['complex'] = np.column_stack(complex)
            else:
                data['complex'] = None

            if 'zpower' in data.keys() and data['zpower'] is not None:
                data['power'] = W.decompress1(data['zpower'], 'db4', level=level, shapes=data['zpshapes'])[:data['count']].astype(np.int16)
                del data['zpshapes']
                del data['zpower']

            del data['zlevel']
            del data['zshapes']
            del data['zcomplex']
        case _:
            assert False, f'Datagram type {data['type']} not supported.'

    return data


def stats(dk, rk, eps=1e-8):
    assert dk.dtype == dk.dtype, f'Type mismatch, original: {dk.dtype}, reconstructed: {rk.dtype}'
    assert dk.shape == rk.shape, f'Shape mismatch, original: {dk.shape}, reconstructed: {rk.shape}'
    diff = dk - rk
    abs_diff = np.abs(diff)
    mae = np.mean(abs_diff)
    mse = np.mean(abs_diff ** 2)
    denom = np.abs(dk)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs(diff / (dk + eps * (denom == 0))))
        mape = np.nan_to_num(mape) * 100
    return mae, mse, mape


def comptest(fname, level, threshold_ratio):
    '''Test compression functionality by compressing and decompressing all RAW datagrams'''
    for dgram in ekfile(fname).datagrams():
        if dgram[0] == 'RAW3':
            data = SimradRawParser().from_string(dgram[3], len(dgram[3]))
            zdata = raw2raz(data, level=level, threshold_ratio=threshold_ratio)
            zd = SimradRawZParser().to_string(zdata)
            zr = SimradRawZParser().from_string(zd[4:], len(zd) - 8)
            rdata = raz2raw(zr)

            # added_fields = []
            # for k in zdata.keys():
            #     if k not in data.keys():
            #         added_fields.append(k)
            # if added_fields: print(f'Compressed data contains new fields {added_fields}.')
            # lost_fields = []
            # for k in data.keys():
            #     if k not in zdata.keys():
            #         lost_fields.append(k)
            # if lost_fields: print(f'Compressed data do not contain fields {lost_fields}.')

            for k in rdata.keys():
                if k not in data.keys():
                    print(f'Warning: found field "{k}" in recovered data, not present in original')
                elif type(data[k]) is not type(rdata[k]):
                    print(f'Warning: field "{k}" does not have correct type: {type(data[k])} vs {type(rdata[k])}')
            for k in data.keys():
                if k not in rdata.keys():
                    print(f'Warning: did not find field "{k}" in recovered data')
            for k in ['complex', 'angles', 'power']:
                if k in data.keys() and data[k] is not None:
                    dsize = data[k].nbytes
                    if k == 'complex':
                        zsize = 0
                        for z in zdata['z' + k]:
                            zsize += sum(len(y) for y in z)
                    else:
                        zsize = zdata['z' + k].nbytes 
                    mae, mse, mape = stats(data[k], rdata[k])
                    print(f'Field:\t{k}\tUncomp:\t{dsize:6}\tComp:\t{zsize:6}\t'
                          f'{100.0 * zsize / dsize:.1f}%\t', end='')
                    print(f'MAE:\t{mae}\tMAPE:\t{mape}%\tMSE:\t{mse}')


def compress(fname, ofile=None, level=3, threshold=0.2):
    '''Process a RAW file and replace RAWx datagrams with RAZx compressed datagrams.'''
    if ofile:
        outfile = open(ofile, 'wb')
    elif fname == '-' or ofile == '-':
        outfile = sys.stdout.buffer
    elif not os.path.exists(fname + '.ekz'):
        outfile = open(fname + '.ekz', 'wb')
    else:
        print('Output file exists?')
        exit()
    for dgram in ekfile(fname).datagrams():
        if dgram[0] == 'RAW3':   # replace with compressed version
            data = SimradRawParser().from_string(dgram[3], len(dgram[3]))
            zdata = raw2raz(data, level=level, threshold_ratio=threshold)
            zd = SimradRawZParser().to_string(zdata)
            outfile.write(zd)
        else:
            dgram_write(outfile, dgram[3])
    if fname != '-': outfile.close()


def decompress(fname, ofile=None):
    '''Process a RAW file and replace RAZx datagrams with RAWx uncompressed datagrams.'''
    if ofile:
        outfile = open(ofile, 'wb')
    elif fname == '-' or ofile == '-':
        outfile = sys.stdout.buffer
    elif fname.endswith('.ekz') and not os.path.exists(fname[:-4]):
        outfile = open(fname[:-4], 'wb')
    else:
        print('Output file exists, or input file has unknown suffix')
        exit()
    for dgram in ekfile(fname).datagrams():
        if dgram[0] == 'RAZ3':
            zdata = SimradRawZParser().from_string(dgram[3], len(dgram[3]))
            rdata = raz2raw(zdata)
            ndgram = SimradRawParser().to_string(rdata)
            outfile.write(ndgram)
        else:
            dgram_write(outfile, dgram[3])
    if fname != '-': outfile.close()


def main():
    parser = argparse.ArgumentParser(description="Compress or decompress Simrad RAW files.")
    parser.add_argument('files', help="Files to process", nargs="*")
    parser.add_argument('-d', '--decompress', action='store_true', help="Decompress file")
    parser.add_argument('-o', help="Output file name")
    parser.add_argument('--statistics', action='store_true', help="Test execution and output statistics")
    # Compression level options:
    parser.add_argument('--level', type=int, choices=range(2, 6), default=3, help='Compression wavelet levels, from 2 to 6')
    parser.add_argument('--threshold', type=float, default=0.2, help='Compression threshold.')

    args = parser.parse_args()
    decompress_mode = args.decompress or 'unzip' in os.path.basename(sys.argv[0])

    if not args.files: args.files = ['-']
    if args.o and len(args.files) != 1:
        print('Error: refusing to compress multiple files when output file is specified')
        exit()
    for f in args.files:
        if decompress_mode:
            decompress(f, args.o)
        elif args.statistics:
            comptest(f, args.level, args.threshold)
        else:
            compress(f, args.o, args.level, args.threshold)


if __name__ == '__main__':
    main()
