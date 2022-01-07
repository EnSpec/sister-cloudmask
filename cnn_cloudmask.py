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
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import datetime as dt

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
    parser.add_argument('obs_file', type=str,
                        help='Input observables image')
    parser.add_argument('out_dir', type=str,
                         help='Output directory')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--median', type=int, default = 3)
    parser.add_argument('--dilation', type=int, default = 9)
    parser.add_argument('--cld_prb', type=float, default = .3)
    parser.add_argument('--cls', action='store_true')

    args = parser.parse_args()

    rdn = ht.HyTools()
    rdn.read_file(args.rdn_file, 'envi')
    rdn.load_data()

    obs = ht.HyTools()
    obs.read_file(args.obs_file, 'envi')
    obs.load_data()

    model_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/'
    data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data/'

    out_dir = args.out_dir+'/' if not args.out_dir.endswith('/') else args.out_dir

    esd = pd.read_csv('%s/earth_sun_distance.csv' % data_dir,index_col=0,header = None)
    solar = pd.read_csv('%s/kurudz_0.1nm.csv' % data_dir,index_col=0,header = None)

    if rdn.base_name.startswith('PRS'):
        model_path  = '%s/prisma_cnn_v1.h5' % model_dir
        model = keras.models.load_model(model_path)
        waves = np.arange(420,2441,10)
    else:
        print("Unrecognized sensor.")
        return

    #Generate solar irradiance spectrum
    cos_sol_zen =np.cos(np.radians(obs.get_band(4)))
    sun_irad = []

    if len(sun_irad) == 0:
        for i,wave in enumerate(rdn.wavelengths):
            sigma = (rdn.fwhm[i]*10)/(2* np.sqrt(2*np.log(2)))
            sol = gaussian_filter1d(solar,sigma).flatten()
            sun_irad.append(sol[np.argwhere(round(wave,1) ==solar.index)[0][0]])
    sun_irad = np.array(sun_irad)

    doy = dt.datetime.strptime(rdn.base_name[4:12],
                               '%Y%m%d').timetuple().tm_yday

    iterator = rdn.iterate(by = 'chunk')
    classes = np.zeros((rdn.lines,rdn.columns,4))

    pixels = 0
    while not iterator.complete:
        chunk = np.copy(iterator.read_next())
        pixels+=chunk.shape[0]*chunk.shape[1]
        chunk[chunk <=0] = .00001

        #Calculate ToA reflectance and interpolate to 10nm
        toa_chunk = np.pi*chunk*(esd.loc[doy].values[0]**2)
        cos_chunk = cos_sol_zen[iterator.current_line:iterator.current_line+chunk.shape[0],
                                 iterator.current_column:iterator.current_column+chunk.shape[1]]
        toa_chunk /= cos_chunk[:,:,np.newaxis]* sun_irad[np.newaxis,np.newaxis,:]

        interper = interp1d(rdn.wavelengths,toa_chunk,
                            kind='linear',fill_value = 'extrapolate')
        #Interpolate data
        toa_chunk = interper(waves).astype(float)
        toa_chunk = toa_chunk.reshape((toa_chunk.shape[0]*toa_chunk.shape[1],
                                       toa_chunk.shape[2]))
        #Smooth data
        toa_chunk = savgol_filter(toa_chunk,7,2)
        pred= model.predict(toa_chunk[:,:,np.newaxis])
        pred = pred.reshape((chunk.shape[0],chunk.shape[1],4))
        classes[iterator.current_line:iterator.current_line+pred.shape[0],
                iterator.current_column:iterator.current_column+pred.shape[1],:] = pred


    clouds = classes[:,:,3]>args.cld_prb
    cloud_mask =median_filter(clouds.astype(bool),args.median)
    cloud_dilate = binary_dilation(cloud_mask,
                                    structure= np.ones((args.dilation,args.dilation)))
    # Export cloud mask
    mask_header = rdn.get_header()
    mask_header['bands']= 1
    mask_header['band names']= ['cloud']
    mask_header['wavelength']= []
    mask_header['fwhm']= []

    out_file = rdn.file_name.replace('_rdn','_cld')
    writer = WriteENVI(out_file,mask_header)
    writer.write_band(cloud_dilate,0)

    #Export class probabilites
    if args.cls:
        mask_header['bands']= 4
        mask_header['band names']= ['land','water','snow_ice','cloud']
        out_file = rdn.file_name.replace('_rdn','_cls')
        writer = WriteENVI(out_file,mask_header)
        for band in range(4):
            writer.write_band(classes[:,:,band],band)

if __name__ == "__main__":
    main()
