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
import datetime as dt
import hytools_lite as ht
from hytools_lite.io.envi import WriteENVI
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation,gaussian_filter1d
from scipy.signal import savgol_filter

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


RED = 15.84675484
NDSI = 0.4335392
NIR_RED = 4.39141129
NIR_GREEN = 1.40519307
NIR_SWIR = 2.27128579

def main():
    ''' Generate cloud mask from spaceborne hyperspectral imagery

    Band ratio and index based cloud masking algorithm derived from:

        HyspIRI Cloud Mask Detection Algorithm Theoretical Basis Document
        Hulley, Glynn C.; Hook, Simon J.
        http://hdl.handle.net/2014/42573



        Excludes thermal bands


    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('rdn_file', type=str,
                        help='Input radiance image')
    parser.add_argument('obs_file', type=str,
                        help='Obserables image')
    parser.add_argument('out_dir', type=str,
                         help='Output directory')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dilation', type=int, default = 4)
    parser.add_argument('--apply', action='store_true')

    args = parser.parse_args()

    radiance = ht.HyTools()
    radiance.read_file(args.rdn_file, 'envi')
    radiance.load_data()

    doy = dt.datetime.strptime(radiance.base_name[4:12],'%Y%m%d').timetuple().tm_yday

    observables = ht.HyTools()
    observables.read_file(args.obs_file, 'envi')
    observables.load_data()
    cos_sol_zen =np.cos(np.radians( observables.get_band(4)))

    data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data/'
    out_dir = args.out_dir+'/' if not args.out_dir.endswith('/') else args.out_dir

    #Load external datasets
    esd = pd.read_csv('%s/earth_sun_distance.csv' % data_dir,index_col=0,header = None)
    extrasolar = pd.read_csv('%s/kurudz_0.1nm.csv' % data_dir,index_col=0,header = None)
    waves = np.arange(420,2441,10)

    #Resample extrasolar irradiance to sensor wavelengths
    sun_irad = []
    for i,wave in enumerate(radiance.wavelengths):
        sigma = (radiance.fwhm[i]*10)/(2* np.sqrt(2*np.log(2)))
        sol = gaussian_filter1d(extrasolar,sigma).flatten()
        sun_irad.append(sol[np.argwhere(round(wave,1) ==extrasolar.index)[0][0]])
    sun_irad =np.array(sun_irad)

    cloud_mask = np.full((radiance.lines,radiance.columns),-1.0)
    iterator =radiance.iterate(by = 'line')
    while not iterator.complete:

        line = iterator.read_next()

        #Calculate TOA reflectance
        toa = np.pi*line*(esd.loc[doy].values[0]**2)
        toa /= cos_sol_zen[[iterator.current_line],:].T* sun_irad[np.newaxis,:]
        toa*=1000
        #Resample to regular interval
        interper = interp1d(radiance.wavelengths,toa,kind='linear',fill_value = 'extrapolate')
        toa = interper(waves).astype(float)
        #Smooth TOA reflectance
        toa = savgol_filter(toa,9,1)

        green_index = np.argwhere(waves == 550)[0][0]
        red_index = np.argwhere(waves == 650)[0][0]
        nir_index = np.argwhere(waves == 800)[0][0]
        swir_index = np.argwhere(waves == 1650)[0][0]

        cloud_mask[iterator.current_line,:] = (toa[:,red_index] > RED).astype(int)
        cloud_mask[iterator.current_line,:] += ((toa[:,green_index]-toa[:,swir_index])/(toa[:,green_index]+toa[:,swir_index]) < NDSI).astype(int)
        cloud_mask[iterator.current_line,:] += (toa[:,nir_index]/toa[:,red_index] < NIR_RED).astype(int)
        cloud_mask[iterator.current_line,:] += (toa[:,nir_index]/toa[:,green_index] < NIR_GREEN).astype(int)
        cloud_mask[iterator.current_line,:] += (toa[:,nir_index]/toa[:,swir_index] < NIR_SWIR).astype(int)

        if args.verbose:
            progbar(iterator.current_line,radiance.lines, full_progbar = 100)
    print('\n')

    y_grid, x_grid = np.ogrid[-args.dilation: args.dilation + 1, -args.dilation: args.dilation + 1]
    window =  (x_grid**2 + y_grid**2 <= args.dilation**2)

    clouds =binary_dilation(cloud_mask==5,
                                structure= window)
    clouds[~radiance.mask['no_data']] = -9999

    # Export cloud radiance
    mask_header = radiance.get_header()
    mask_header['bands']= 1
    mask_header['band names']= ['cloud']
    mask_header['wavelength']= []
    mask_header['fwhm']= []
    mask_header['data type']= 4
    mask_header['default bands']= []

    out_file = out_dir + radiance.base_name + '_cloud'
    writer = WriteENVI(out_file,mask_header)
    writer.write_band(clouds,0)

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
