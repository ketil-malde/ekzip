# import numpy as np
import sys
import struct
from ektools import ekfile  # , parse
from simrad_parsers import SimradRawParser
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
                    complex.append(zd)  # todo: clip to n_complex?
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


def comptest(fname):
    '''Test compression functionality by compressing and decompressing all RAW datagrams'''
    for dgram in ekfile(fname).datagrams():
        if dgram[0] == 'RAW3':
            data = SimradRawParser().from_string(dgram[3], len(dgram[3]))
            zdata = raw2raz(data)
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
                    s1 = data[k].shape
                    s2 = rdata[k].shape
                    t1 = data[k].dtype
                    t2 = rdata[k].dtype
                    if s1 != s2: print(f'Shape of field {k} changed from {s1} to {s2}')
                    if t1 != t2: print(f'Type of field {k} changed from {t1} to {t2}')
                    # print(f'Before: {data[k][:20]}')
                    # print(f'After: {rdata[k][:20]}')
                    absdiffs = np.abs(data[k] - rdata[k])  # WTF?  Adding: .astype(float)  changes MSE?
                    compr = len(zdata["z" + k]) / len(data[k])
                    print(f'Field:\t{k}\tUncomp:\t{len(data[k]):6}\tComp:\t{len(zdata["z"+k]):6}\t{100 * compr:.1f}%\t', end='')
                    print(f'MAE:\t{np.mean(absdiffs):.1f}\tMAPE:\t{np.mean(100 * np.abs((data[k] - rdata[k]) / data[k])):.1f}%\tMSE:\t{np.mean(absdiffs**2):.1f}')


def compress(fname):
    '''Process a RAW file and replace RAWx datagrams with RAZx compressed datagrams.'''
    with open(fname + '.ekz', 'wb') as outfile:
        for dgram in ekfile(fname).datagrams():
            if dgram[0] == 'RAW3':   # replace with compressed version
                data = SimradRawParser().from_string(dgram[3], len(dgram[3]))
                zdata = raw2raz(data)
                zd = SimradRawZParser().to_string(zdata)
                outfile.write(zd)
            else:
                dgram_write(outfile, dgram[3])


def decompress(fname):
    '''Process a RAW file and replace RAZx datagrams with RAWx uncompressed datagrams.'''
    with open(fname + '.new', 'wb') as outfile:
        for dgram in ekfile(fname + '.z').datagrams():
            if dgram[0] == 'RAZ3':
                print('decompressing:', dgram[:3])
                zdata = SimradRawZParser().from_string(dgram[3], len(dgram[3]))
                rdata = raz2raw(zdata)
                ndgram = SimradRawParser().to_string(rdata)
                outfile.write(ndgram)
            else:
                dgram_write(outfile, dgram[3])

# todo: implement range clipping and bottom detection?


def main():
    # argparser
    parser = argparse.ArgumentParser(description="Compress or decompress Simrad RAW files.")
    parser.add_argument('files', help="Files to process", nargs="+")
    parser.add_argument('-d', '--decompress', action='store_true', help="Decompress file")
    parser.add_argument('--debug', action='store_true', help="Test and output statistics")

    args = parser.parse_args()
    print('args', args)
    decompress_mode = args.decompress or 'unzip' in os.path.basename(sys.argv[0])

    for f in args.files:
        print(f)
        if decompress_mode:
            decompress(f)
        elif args.debug:
            comptest(f)
        else:
            compress(f)


if __name__ == '__main__':
    main()
