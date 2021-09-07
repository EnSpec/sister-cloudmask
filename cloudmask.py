"""
SISTER
Space-based Imaging Spectroscopy and Thermal PathfindER
Author: Adam Chlus
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import os
import hytools_lite as ht
from hytools_lite.io.envi import WriteENVI
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation,median_filter
from skimage.util import view_as_blocks
from tensorflow import keras

def progbar(curr, total, full_progbar = 100):
    '''Display progress bar.
    Gist from:
    https://gist.github.com/marzukr/3ca9e0a1b5881597ce0bcb7fb0adc549

    Args:
        curr (int, float): Current task level.
        total (int, float): Task level at completion.
        full_progbar (TYPE): Defaults to 100.
    Returns:
        None.
    '''
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')

def main():
    ''' Generate cloud mask from spaceborne hyperspectral imagery
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('rdn_file', type=str,
                        help='Input radiance image')
    parser.add_argument('out_dir', type=str,
                         help='Output directory')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--median', type=int, default = 7)
    parser.add_argument('--dilation', type=int, default = 7)
    parser.add_argument('--apply', action='store_true')

    args = parser.parse_args()

    radiance = ht.HyTools()
    radiance.read_file(args.rdn_file, 'envi')
    radiance.load_data()

    model_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/'
    out_dir = args.out_dir+'/' if not args.out_dir.endswith('/') else args.out_dir

    if (radiance.wavelengths.max() >= 990) and (radiance.wavelengths.max() < 2440):
        waves = np.arange(420,981,10)
        model_path  = '%s/cloud_rad_vnir.h5' % model_dir
    elif radiance.wavelengths.max() > 2440:
        waves = np.arange(420,2451,10)
        model_path  = '%s/cloud_rad_vswir.h5' % model_dir
    else:
        print("Image wavelength range outside of model ranges.")
        return

    #Calculate wavelengths aggregation bins
    bins = int(np.round(10/np.diff(radiance.wavelengths).mean()))
    agg_waves  = np.nanmean(view_as_blocks(radiance.wavelengths[:(radiance.bands//bins) * bins],
                                            (bins,)),axis=1)
    model = keras.models.load_model(model_path)

    cloud_mask = np.full((radiance.lines,radiance.columns),-1.0)
    iterator =radiance.iterate(by = 'chunk',chunk_size = (200,200))
    i = 0
    while not iterator.complete:

        chunk = iterator.read_next()
        chunk =view_as_blocks(chunk[:,:,:(radiance.bands//bins) * bins],
                              (1,1,bins)).mean(axis=(-3,-2,-1))
        data =chunk.reshape(chunk.shape[0]*chunk.shape[1],chunk.shape[2])
        interper = interp1d(agg_waves,data,kind='cubic')
        data = interper(waves)
        chunk_prd = model.predict(data[:,:,np.newaxis])
        pred =np.argmax(chunk_prd,axis=1).reshape(chunk.shape[:2])
        cloud_mask[iterator.current_line:iterator.current_line+chunk.shape[0],
              iterator.current_column:iterator.current_column+chunk.shape[1]] = pred
        i+=pred.shape[0]*pred.shape[1]
        if args.verbose:
            progbar(i,radiance.lines*radiance.columns, full_progbar = 100)
    print('\n')

    # Apply spatial filters to classification map
    cloud_mask =median_filter(cloud_mask,args.median)
    labels = cloud_mask==3
    labels_dilate = binary_dilation(labels,
                                    structure= np.ones((args.dilation,
                                                        args.dilation)) ==1)
    cloud_mask[labels_dilate] = 3
    cloud_mask[~radiance.mask['no_data']] = -9999

    # Export cloud radiance
    mask_header = radiance.get_header()
    mask_header['bands']= 1
    mask_header['band names']= ['cover_class']
    mask_header['wavelength']= []
    mask_header['fwhm']= []
    mask_header['data type']= 2
    out_file = out_dir + radiance.base_name + '_cls'
    writer = WriteENVI(out_file,mask_header)
    writer.write_band(cloud_mask,0)

    if args.apply:
        # Export masked radiance
        out_header = radiance.get_header()
        out_file = out_dir + radiance.base_name + '_msk'
        writer = WriteENVI(out_file,out_header)

        mask = cloud_mask == 1

        for band_num in range(radiance.bands):
            band = np.copy(radiance.get_band(band_num))
            band[mask] = radiance.no_data
            writer.write_band(band,band_num)

if __name__ == "__main__":
    main()
