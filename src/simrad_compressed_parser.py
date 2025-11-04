# The RAW parser, only with data compression
import struct
import numpy as np
from ektools.simrad_parsers import _SimradDatagramParser
from ektools.date_conversion import nt_to_unix


class SimradRawZParser(_SimradDatagramParser):
    '''
    Sample Data Datagram parser operates on dictonaries with the following keys:

        type:         string == 'RAZ0'
        low_date:     long uint representing LSBytes of 64bit NT date
        high_date:    long uint representing MSBytes of 64bit NT date
        timestamp:    datetime.datetime object of NT date, assumed to be UTC

        channel                         [short] Channel number
        mode                            [short] 1 = Power only, 2 = Angle only 3 = Power & Angle
        transducer_depth                [float]
        frequency                       [float]
        transmit_power                  [float]
        pulse_length                    [float]
        bandwidth                       [float]
        sample_interval                 [float]
        sound_velocity                  [float]
        absorption_coefficient          [float]
        heave                           [float]
        roll                            [float]
        pitch                           [float]
        temperature                     [float]
        heading                         [float]
        transmit_mode                   [short] 0 = Active, 1 = Passive, 2 = Test, -1 = Unknown
        spare0                          [str]
        offset                          [long]
        count                           [long]

        power                           [numpy array] Unconverted power values (if present)
        angle                           [numpy array] Unconverted angle values (if present)

    from_string(str):   parse a raw sample datagram
                        (with leading/trailing datagram size stripped)

    to_string(dict):    Returns raw string (including leading/trailing size fields)
                        ready for writing to disk
    '''

    def __init__(self):
        headers = {0 : [('type', '4s'),
                        ('low_date', 'L'),
                        ('high_date', 'L'),
                        ('channel', 'h'),
                        ('mode', 'h'),
                        ('transducer_depth', 'f'),
                        ('frequency', 'f'),
                        ('transmit_power', 'f'),
                        ('pulse_length', 'f'),
                        ('bandwidth', 'f'),
                        ('sample_interval', 'f'),
                        ('sound_velocity', 'f'),
                        ('absorption_coefficient', 'f'),
                        ('heave', 'f'),
                        ('roll', 'f'),
                        ('pitch', 'f'),
                        ('temperature', 'f'),
                        ('heading', 'f'),
                        ('transmit_mode', 'h'),
                        ('spare0', '6s'),
                        ('offset', 'l'),
                        ('count', 'l')
                        ],
                   3 : [('type', '4s'),
                        ('low_date', 'L'),
                        ('high_date', 'L'),
                        ('channel_id', '128s'),
                        ('data_type', 'h'),
                        ('spare', '2s'),
                        ('offset', 'l'),
                        ('count', 'l')
                        ],
                    4 : [('type', '4s'),
                        ('low_date', 'L'),
                        ('high_date', 'L'),
                        ('channel_id', '128s'),
                        ('data_type', 'h'),
                        ('spare', '2s'),
                        ('offset', 'l'),
                        ('count', 'l')
                        ]
                    }
        _SimradDatagramParser.__init__(self, 'RAZ', headers)

    def _unpack_contents(self, raw_string, bytes_read, version):

        header_values = struct.unpack(self.header_fmt(version), raw_string[:self.header_size(version)])

        data = {}

        for indx, field in enumerate(self.header_fields(version)):
            data[field] = header_values[indx]
            if isinstance(data[field], bytes):
                #  first try to decode as utf-8 but fall back to latin_1 if that fails
                try:
                    data[field] = data[field].decode("utf-8")
                except:
                    data[field] = data[field].decode("latin_1")

        data['timestamp'] = nt_to_unix((data['low_date'], data['high_date']))
        data['timestamp'] = data['timestamp'].replace(tzinfo=None)
        data['bytes_read'] = bytes_read

        if version == 0:

            if data['count'] > 0:
                block_size = data['count'] * 2
                indx = self.header_size(version)

                if int(data['mode']) & 0x1:
                    data['power'] = np.frombuffer(raw_string[indx:indx + block_size], dtype='int16')
                    indx += block_size
                else:
                    data['power'] = None

                if int(data['mode']) & 0x2:
                    data['angle'] = np.frombuffer(raw_string[indx:indx + block_size], dtype='int8')
                    data['angle'].shape = (data['count'], 2)
                else:
                    data['angle'] = None

            else:
                data['power'] = np.empty((0,), dtype='int16')
                data['angle'] = np.empty((0,), dtype='int8')

        elif version == 3 or version == 4:

            #  clean up the channel ID
            data['channel_id'] = data['channel_id'].strip('\x00')

            if data['count'] > 0:

                #  set the initial block size and indx value.
                block_size = data['count'] * 2
                indx = self.header_size(version)

                if data['data_type'] & 0b1:
                    zpowershapes = struct.unpack('i', raw_string[indx:indx + 4])[0]
                    indx += 4
                    data['zpshapes'] = struct.unpack("%di" % zpowershapes, raw_string[indx:indx + 4 * zpowershapes])
                    indx += 4 * zpowershapes

                    zpowerlen = struct.unpack('i', raw_string[indx:indx + 4])[0]
                    # print('..unpacked zplen:', zpowerlen, 'zpshapes:', data['zpshapes'])

                    indx += 4
                    data['zpower'] = raw_string[indx:indx + zpowerlen]
                    indx += zpowerlen
                    # data['power'] = np.frombuffer(raw_string[indx:indx + block_size], dtype='int16')
                    # indx += block_size
                else:
                    data['power'] = None

                if data['data_type'] & 0b10:
                    data['angle'] = np.frombuffer(raw_string[indx:indx + block_size], dtype='int8')
                    data['angle'].shape = (data['count'], 2)
                    indx += block_size
                else:
                    data['angle'] = None

                #  determine the complex sample data type - this is contained in bits 2 and 3
                #  of the datatype <short> value. I'm assuming the types are exclusive...
                #  Note that Numpy doesn't support the complex32 type so both the full precision
                #  (complex comprised of 2 32-bit floats) and reduced precision (complex
                #  comprised of 2 16-bit floats) are returned as np.complex64 which is complex
                #  comprised of 2 32-bit floats.
                if ((data['data_type'] & 0b1000)):
                    data['complex_dtype'] = np.float32
                else:
                    data['complex_dtype'] = np.float16

                #  determine the number of complex samples
                data['n_complex'] = data['data_type'] >> 8

                #  unpack the compressed complex samples
                if (data['n_complex'] > 0):
                    # read level parameter and lenght of shapes array
                    data['zlevel'], zshapelen = struct.unpack('ii', raw_string[indx:indx + 8])
                    indx += 8
                    # read the zshapes array
                    data['zshapes'] = struct.unpack("%di" % zshapelen, raw_string[indx:indx + 4 * zshapelen])
                    indx += 4 * zshapelen

                    # read zcomplex vectors (real and imag)
                    zcomplex = []
                    for i in range(data['n_complex']):
                        zc = []
                        for j in [0, 1]:
                            zlen = struct.unpack('i', raw_string[indx:indx + 4])[0]
                            indx += 4
                            zc.append(raw_string[indx:indx + zlen])
                            indx += zlen
                        zcomplex.append((zc[0], zc[1]))
                    data['zcomplex'] = zcomplex
                else:
                    data['zcomplex'] = None
                    data['zlevel'] = None
                    data['zshapes'] = None
            else:
                # Does this make sense here?  If count is zero...then what?  Why not None, like above?
                data['power'] = np.empty((0,), dtype='int16')
                data['angle'] = np.empty((0,), dtype='int8')
                data['zcomplex'] = None
                data['zlevel'] = None
                data['zshapes'] = None
                data['n_complex'] = 0

        return data

    def _pack_contents(self, data, version):

        datagram_fmt = self.header_fmt(version)
        datagram_contents = []

        if version == 0:

            if data['count'] > 0 and data['mode'] == 0:
                data['count'] = 0

            for field in self.header_fields(version):
                if isinstance(data[field], str):
                    data[field] = data[field].encode('latin_1')
                datagram_contents.append(data[field])

            if data['count'] > 0:

                if int(data['mode']) & 0x1:
                    datagram_fmt += '%dh' % (data['count'])
                    datagram_contents.append(data['power'])

                if int(data['mode']) & 0x2:
                    n_angles = data['count'] * 2
                    datagram_fmt += '%db' % (n_angles)
                    #  reshape the angle array for writing
                    data['angle'].shape = (n_angles,)
                    datagram_contents.extend(data['angle'])

        elif version == 3 or version == 4:

            # Add the spare field
            data['spare'] = ''

            # work through the parameter dict and append data values to the
            # packed datagram list.
            for field in self.header_fields(version):
                if isinstance(data[field], str):
                    data[field] = data[field].encode('latin_1')
                datagram_contents.append(data[field])

            # Check if we have data to write
            if data['count'] > 0:

                if data['data_type'] & 0b0001:
                    zpowershapes = len(data['zpshapes'])
                    datagram_fmt += 'i%di' % zpowershapes
                    for d in [zpowershapes, *data['zpshapes']]:
                        datagram_contents.append(d)

                    zpowerlen = len(data['zpower'])
                    # print('..packing zplen:', zpowerlen, 'shapes:', data['zpshapes'])
                    datagram_fmt += 'i%dB' % zpowerlen
                    datagram_contents.append(zpowerlen)
                    datagram_contents.extend(data['zpower'])

                if data['data_type'] & 0b0010:
                    # Add the angle data
                    n_angles = data['count'] * 2
                    datagram_fmt += '%db' % (n_angles)
                    #  reshape the angle array for writing
                    data['angle'].shape = (n_angles,)
                    datagram_contents.extend(data['angle'])

                if data['data_type'] & 0b1100:
                    # Write the compressed complex data
                    if data['data_type'] & 0b0100:
                        # This shouldn't matter, our zcomplex field is written as it is.
                        # pack as 16 bit floats - struct doesn't have support for
                        # half floats so we use just pack them as bytes.
                        # datagram_fmt += '%dB' % (data['complex'].shape[0] * 2 * 8)
                        assert False, 'not implemented here'
                    else:
                        # storing level, len shapes, and the shapes
                        datagram_fmt += 'ii%di' % len(data['zshapes'])
                        for d in [data['zlevel'], len(data['zshapes']), *data['zshapes']]:
                            datagram_contents.append(d)
                        for i in range(data['n_complex']):
                            zdata = data['zcomplex'][i]
                            for j in [0, 1]:  # real and imag
                                zlen = len(zdata[j])
                                datagram_fmt += 'i%dB' % zlen
                                datagram_contents.append(zlen)
                                datagram_contents.extend(zdata[j])

                        # datagram_fmt += '%dB' % (data['complex'].shape[0] * 4 * 8)
                    # outbytes = data['complex'].tobytes()  # view(np.ubyte)
                    # print('Packing, shape, size, len:', outbytes.shape, outbytes.size, len(outbytes))
                    # datagram_contents.extend(outbytes)

        return struct.pack(datagram_fmt, *datagram_contents)
