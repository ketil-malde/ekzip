# import numpy as np
import sys
import struct
from ektools import ekfile  # , parse
from simrad_parsers import SimradRawParser
from simrad_compressed_parser import SimradRawZParser
import numpy as np

import wavelets as W

# or just use dgram_write from eksplit?  Or finalize_datagram?
def dgram_write(f, dgram):
    """Write a datagram to a file"""
    hdr = struct.pack('<l', len(dgram))
    f.write(hdr)
    f.write(dgram)
    f.write(hdr)


def raw2raz(data):  # (dgram):
    '''Convert the dict representing a RAW type datagram into a RAZ compressed datatgram'''
    print(data)
    match data['type']:
        case 'RAW3':
            data['type'] = 'RAZ' + data['type'][3]

            if data['n_complex'] > 0:
                zcomplex = []
                for i in range(data['n_complex']):
                    zd, wl, lv, sh = W.compress(data['complex'][:, i], wavelet='db4', level=3, threshold_ratio=0.2)
                    zcomplex.append(zd)  # oh fuck, it's a tuple, real/imag

                data['zlevel'] = lv
                data['zshapes'] = [s[0] for s in sh]
                data['zcomplex'] = zcomplex
                del data['complex']
        case _:
            assert False, f'Datagram type {data['type']} not supported.'

    return data


def raz2raw(data):  # dgram:
    '''Convert the dict representing a RAZ compressed datagram into a RAW datatgram'''
    # data = SimradRawZParser().from_string(dgram, len(dgram))
    match data['type']:
        case 'RAZ3':
            data['type'] = 'RAW' + data['type'][3]

            if data['n_complex'] > 0:
                level = data['zlevel']
                shapes = [(s,) for s in data['zshapes']]
                complex = []
                for i in range(data['n_complex']):
                    zd = W.decompress(data['zcomplex'][i], 'db4', level=level, shapes=shapes)
                    complex.append(zd)
                data['complex'] = np.column_stack(complex)
            else:
                data['complex'] = None
            del data['zlevel']
            del data['zshapes']
            del data['zcomplex']
        case _:
            assert False, f'Datagram type {data['type']} not supported.'

    return data


def compress(fname, ):
    '''Process a RAW file and replace RAWx datagrams with RAZx compressed datagrams.'''
    with open(fname + '.z', 'wb') as outfile:
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
                zdata = SimradRawZParser().from_string(dgram[3], len(dgram[3]))
                rdata = raz2raw(zdata)
                ndgram = SimradRawParser().to_string(rdata)
                outfile.write(ndgram)
            else:
                dgram_write(outfile, dgram[3])


if __name__ == '__main__':
    for f in sys.argv[1:]:
        compress(f)
        decompress(f)


''' Old code:

                data = SimradRawParser().from_string(dgram[3], len(dgram[3]))
                # data = raw2raz(data)
                newdg = SimradRawZParser().to_string(data)
                try:
                    data2 = SimradRawParser().from_string(newdg[4:-4], len(newdg)-8)  # should fail
                except Exception as e:
                    pass
                data2 = SimradRawZParser().from_string(newdg[4:-4], len(newdg)-8)  # should work
'''
