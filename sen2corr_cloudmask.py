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
from scipy.ndimage import gaussian_filter1d

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



SENT_BANDS = [443, 490,
              560,
              665,
              705, 740, 783,
              842,865,
              945,
              1375,
              1610,
              2190]

CLASS_CODES = {'dark':0,
               'cloud_shadow':1,
               'veg':2,
               'soil':3,
               'water':4,
               'cloud_low':5,
               'cloud_med':6,
               'cloud_high':7,
               'cirrus':8,
               'snow':9}



def main():
    ''' Generate cloud mask from spaceborne hyperspectral imagery

    Band ratio and index based cloud masking algorithm derived from:

    Sentinel-2 MSI – Level 2A Products Algorithm Theoretical Basis Document
    R. Richter (DLR), J. Louis, B. Berthelot (VEGA France)


    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('rdn_file', type=str,
                        help='Input radiance image')
    parser.add_argument('obs_file', type=str,
                        help='Obserables image')
    parser.add_argument('out_dir', type=str,
                         help='Output directory')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--sentinel', action='store_true')

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
    sentinel_srf = pd.read_csv('%s/sentinel2a_srf.csv' % data_dir,index_col=0)
    #Normalized sentinel SRFs
    sentinel_srf/=sentinel_srf.sum()

    waves = np.arange(412,2321)

    #Resample extrasolar irradiance to sensor wavelengths
    sun_irad = []
    for i,wave in enumerate(radiance.wavelengths):
        sigma = (radiance.fwhm[i]*10)/(2* np.sqrt(2*np.log(2)))
        sol = gaussian_filter1d(extrasolar,sigma).flatten()
        sun_irad.append(sol[np.argwhere(round(wave,1) ==extrasolar.index)[0][0]])
    sun_irad =np.array(sun_irad)

    sentinel = np.zeros((radiance.lines,
                           radiance.columns,
                           sentinel_srf.shape[1]))

    iterator =radiance.iterate(by = 'line')
    if args.verbose:
        print('Resampling to Sentinel bands')

    while not iterator.complete:

        line = iterator.read_next()

        #Calculate TOA reflectance
        toa = np.pi*line*(esd.loc[doy].values[0]**2)
        toa /= cos_sol_zen[[iterator.current_line],:].T* sun_irad[np.newaxis,:]
        toa*=10
        #Resample to regular interval
        interper = interp1d(radiance.wavelengths,toa,kind='linear',fill_value = 'extrapolate')
        toa = interper(waves).astype(float)

        for s,sen_band in enumerate(sentinel_srf.columns):
            sen_resample = (sentinel_srf.loc[waves,sen_band].values[np.newaxis,:]*toa).sum(axis=1)
            sen_resample[line[:,0] == radiance.no_data] = radiance.no_data
            sentinel[iterator.current_line,:,s] = sen_resample

        if args.verbose:
            progbar(iterator.current_line,radiance.lines, full_progbar = 100)
    print('\n')

    class_map =  np.full((radiance.lines,radiance.columns),-1)
    clouds = np.zeros((radiance.lines,radiance.columns))

    #Step 1a - Brightness thresholds on red (Band 4)
    s1a_t1 = 0.07
    s1a_t2 = 0.25
    s1a_conf = np.clip((sentinel[:,:,3]-s1a_t1)/(s1a_t2-s1a_t1),0,1)
    clouds = s1a_conf

    #Step 1b – Normalized Difference Snow Index (NDSI)
    s1b_t1 = -0.24
    s1b_t2 = 0.16
    ndsi = (sentinel[:,:,2]-sentinel[:,:,11])/(sentinel[:,:,2]+sentinel[:,:,11])
    s1b_conf = np.clip((ndsi-s1b_t1)/(s1b_t2-s1b_t1),0,1)
    clouds[(s1b_conf!=0) | (s1b_conf!=1)] = clouds[(s1b_conf!=0) | (s1b_conf!=1)]* s1b_conf[(s1b_conf!=0) | (s1b_conf!=1)]

    snow = np.zeros((radiance.lines,radiance.columns))
    snow[(clouds!=0)&(clouds!=1)] = clouds[(clouds!=0)&(clouds!=1)]

    # Snow filter 1: Normalized Difference Snow Index (NDSI)
    sf1_t1 = 0.2
    sf1_t2 = 0.42
    sf1_conf = np.clip((ndsi-sf1_t1)/(sf1_t2-sf1_t1),0,1)
    snow = sf1_conf

    # Snow filter 2: Band 8 thresholds
    sf2_t1 = 0.15
    sf2_t2 = 0.35
    sf2_conf = np.clip((sentinel[:,:,7]-sf2_t1)/(sf2_t2-sf2_t1),0,1)
    snow[(snow!=0)&(snow!=1)] = sf2_conf[(snow!=0)&(snow!=1)]

    # Snow filter 3: Band 2 thresholds
    sf3_t1 = 0.18
    sf3_t2 = 0.22
    sf3_conf = np.clip((sentinel[:,:,1]-sf3_t1)/(sf3_t2-sf3_t1),0,1)
    snow *=sf3_conf

    # Snow filter 4: Ratio Band 2 / Band 4
    sf4_t1 = 0.85
    sf4_t2 = 0.95
    b2_b4_ratio =  sentinel[:,:,1]/sentinel[:,:,4]
    sf4_conf = np.clip((b2_b4_ratio-sf4_t1)/(sf4_t2-sf4_t1),0,1)
    snow *=sf4_conf

    #Snow filter 5: Processing of snow boundaries zones
    #Skip for now

    #Update class map to add snow
    class_map[snow ==1] = CLASS_CODES['snow']

    #Step 3 – Normalized Difference Vegetation Index (NDVI)
    s3_t1 = 0.36
    s3_t2 = 0.40
    ndvi =  (sentinel[:,:,7]-sentinel[:,:,3])/(sentinel[:,:,3]+sentinel[:,:,3])
    s3_conf = 1-np.clip((ndvi-s3_t1)/(s3_t2-s3_t1),0,1)
    clouds *= s3_conf

    #Update class map to add vegetation
    class_map[s3_conf == 0] = CLASS_CODES['veg']

    #Step 4 – Ratio Band 8 / Band 3 for senescing vegetation
    s4_t1 = 1.5
    s4_t2 = 2.5
    b8_b3_ratio = sentinel[:,:,7]/sentinel[:,:,2]
    s4_conf = 1-np.clip((b8_b3_ratio-s4_t1)/(s4_t2-s4_t1),0,1)
    clouds *= s4_conf

    #Update class map to add vegetation
    class_map[s4_conf == 0] = CLASS_CODES['veg']

    #Step 5 – Ratio Band 2 / Band 11 for soils and water bodies
    #Pass 1 for soils detection
    s5p1_t1 = .55
    s5p1_t2 = .80
    b2_b11_ratio = sentinel[:,:,1]/sentinel[:,:,11]
    s5p1_conf = np.clip((b2_b11_ratio-s5p1_t1)/(s5p1_t2-s5p1_t1),0,1)
    clouds *= s5p1_t1

    #Update class map to add soil
    class_map[(class_map == -1) & (s5p1_conf == 0)] = CLASS_CODES['soil']

    #Pass 2 for water bodies detection
    s5p2_t1 = 1  #was 2
    s5p2_t2 = 2.5  # was 4
    s5p1_conf = 1-np.clip((b2_b11_ratio-s5p2_t1)/(s5p2_t2-s5p2_t1),0,1)

    clouds *= s5p1_conf

    #Update class map to add water
    class_map[(class_map == -1) & (s5p1_conf == 0)] = CLASS_CODES['water']

    #Step 6 – Ratio Band 8 / band 11 for rocks and sands in deserts
    s6_t1 = .90
    s6_t2 = 1.1
    b8_b11_ratio = sentinel[:,:,7]/sentinel[:,:,11]
    s6_conf = np.clip((b8_b11_ratio-s6_t1)/(s6_t2-s6_t1),0,1)
    clouds *= s6_conf

    #Update class map to add water
    class_map[(class_map == -1) & (s6_conf == 0)] = CLASS_CODES['soil']

    #Sentinel-2 band 10 (1.38 m) thresholds
    cirr_t1 = 0.010
    cirr_t2 = 0.035
    b8_b11_ratio = sentinel[:,:,7]/sentinel[:,:,11]
    cirrus = np.clip((sentinel[:,:,10]-cirr_t1)/(cirr_t2-cirr_t1),0,1)

    #Update class map to add cirrus and clouds
    class_map[cirrus> 0] = CLASS_CODES['cirrus']
    class_map[clouds>0.0] = CLASS_CODES['cloud_low']
    class_map[clouds>0.35] = CLASS_CODES['cloud_med']
    class_map[clouds>0.65] = CLASS_CODES['cloud_high']


    # Export cloud radiance
    mask_header = radiance.get_header()
    mask_header['bands']= 1
    mask_header['band names']= ['class']
    mask_header['wavelength']= []
    mask_header['fwhm']= []
    mask_header['data type']= 4
    mask_header['default bands']= []

    out_file = out_dir + radiance.base_name + '_cloud'
    writer = WriteENVI(out_file,mask_header)
    class_map[~radiance.mask['no_data']] = radiance.no_data
    writer.write_band(class_map,0)

    # if args.apply:
    #     # Export masked radiance
    #     out_header = radiance.get_header()
    #     out_file = out_dir + radiance.base_name + '_msk'
    #     writer = WriteENVI(out_file,out_header)

    #     mask = cloud_mask == 1

    #     for band_num in range(radiance.bands):
    #         band = np.copy(radiance.get_band(band_num))
    #         band[mask] = radiance.no_data
    #         writer.write_band(band,band_num)

    if args.sentinel:
        # Export sentinel resmapled TOA reflectance

        out_header = radiance.get_header()
        out_header['bands'] = sentinel.shape[2]
        out_header['wavelength'] = SENT_BANDS
        out_file = out_dir + radiance.base_name + '_sentinel2'
        writer = WriteENVI(out_file,out_header)

        for band_num in range(sentinel.shape[2]):
            writer.write_band(sentinel[:,:,band_num],band_num)

if __name__ == "__main__":
    main()
