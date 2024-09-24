############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, 2013               #
############################################################


import numpy as np

from mintpy.objects import sensor
from mintpy.utils import ptime, readfile, utils as ut, writefile


################################################################################
def get_corners(atr):
    """Get corners coordinate."""
    length = int(atr['LENGTH'])
    width = int(atr['WIDTH'])
    W = float(atr['X_FIRST'])
    N = float(atr['Y_FIRST'])
    lon_step = float(atr['X_STEP'])
    lat_step = float(atr['Y_STEP'])
    S = N + lat_step * length
    E = W + lon_step * width

    return S, N, W, E, width, length


def get_aoi_lalo(atr_list):
    atr_Alist = []
    atr_Dlist = []
    for i, atr in enumerate(atr_list):
        if atr['ORBIT_DIRECTION'].lower().startswith('asc'):
            atr_Alist.append(atr)
        elif atr['ORBIT_DIRECTION'].lower().startswith('desc'):
            atr_Dlist.append(atr)

    if len(atr_Alist)==1 and len(atr_Dlist)==1:
        print(':: Only 1 Asc and 1 Dsc datasets')
        S, N, W, E = get_overlap_lalo(atr_list)
    else:
        print(':: More than 1 Asc and 1 Dsc datasets')
        Sa, Na, Wa, Ea = get_union_lalo(atr_Alist)
        Sd, Nd, Wd, Ed = get_union_lalo(atr_Dlist)
        W, E = max(Wa, Wd), min(Ea, Ed)
        S, N = max(Sa, Sd), min(Na, Nd)

    return S, N, W, E


def get_union_lalo(atr_list):
    S, N, W, E = None, None, None, None
    for i, atr in enumerate(atr_list):
        Si, Ni, Wi, Ei = ut.four_corners(atr)

        if i == 0:
            S, N, W, E = Si, Ni, Wi, Ei
        else:
            S = min(Si, S)
            N = max(Ni, N)
            W = min(Wi, W)
            E = max(Ei, E)

    return S, N, W, E


def get_overlap_lalo(atr_list):
    """Find overlap area in lat/lon of geocoded files based on their metadata.
    Parameters: atr_list - list of dict, attribute dictionary of two input files in geo coord
    Returns:    S/N/W/E  - float, West/East/South/North in deg
    """
    S, N, W, E = None, None, None, None
    for i, atr in enumerate(atr_list):
        Si, Ni, Wi, Ei = ut.four_corners(atr)
        if i == 0:
            S, N, W, E = Si, Ni, Wi, Ei
        else:
            S = max(Si, S)
            N = min(Ni, N)
            W = max(Wi, W)
            E = min(Ei, E)

    return S, N, W, E


def get_design_matrix4east_north_up(los_inc_angle, los_az_angle, obs_direction=None):
    """Design matrix G to convert multi-track range/azimuth displacement into east/north/up direction.
    Parameters: los_inc_angle - 1D np.ndarray in size of (num_obs,) in float32, LOS incidence angle in degree
                los_az_angle  - 1D np.ndarray in size of (num_obs,) in float32, LOS azimuth   angle in degree
                obs_direction - 1D np.ndarray in size of (num_obs,) in str, observation direction: range or azimuth
    Returns:    G             - 2D np.ndarray in size of (num_obs, 3) in float32, design matrix
    """
    num_obs = los_inc_angle.shape[0]
    G = np.zeros((num_obs, 3), dtype=np.float32)

    # obs_direction: default value
    if not obs_direction:
        obs_direction = ['range'] * num_obs

    # obs_direction: check var type
    if not isinstance(obs_direction, (list, np.ndarray)):
        raise ValueError(f'input obs_direction ({obs_direction}) is NOT a list or numpy.ndarray!')

    for i, (inc_angle, az_angle, obs_dir) in enumerate(zip(los_inc_angle, los_az_angle, obs_direction)):
        # calculate the unit vector
        if obs_dir == 'range':
            # for range offset / InSAR phase [with positive value for motion toward the satellite]
            ve = np.sin(np.deg2rad(inc_angle)) * np.sin(np.deg2rad(az_angle)) * -1
            vn = np.sin(np.deg2rad(inc_angle)) * np.cos(np.deg2rad(az_angle))
            vu = np.cos(np.deg2rad(inc_angle))

        elif obs_dir == 'azimuth':
            # for azimuth offset [with positive value for motion same as flight]
            ve = np.sin(np.deg2rad(az_angle - 90)) * -1
            vn = np.cos(np.deg2rad(az_angle - 90)) * 1
            vu = 0.

        else:
            raise ValueError(f'un-recognized observation direction: {obs_dir}')

        # fill the design matrix
        G[i, :] = [ve, vn, vu]

    return G


def get_design_matrix4no_vert(los_inc_angle, los_az_angle):
    num_file = los_inc_angle.shape[0]
    G = np.zeros((num_file, 2), dtype=np.float32)
    for i in range(num_file):
        G[i, 0] = -np.sin(np.deg2rad(los_inc_angle[i])) * np.sin(np.deg2rad(los_az_angle[i]))
        G[i, 1] =  np.sin(np.deg2rad(los_inc_angle[i])) * np.cos(np.deg2rad(los_az_angle[i]))

    # mute G entries for zero incidence angle (no data there, zero is the default missing pixel)
    G[los_inc_angle==0, :] = 0
    return G


def get_design_matrix4horz_vert(los_inc_angle, los_az_angle, horz_az_angle=-90):
    """Design matrix G to convert asc/desc range displacement into horz/vert direction.
    Only asc + desc -> hz + up is implemented for now.

    Project displacement from LOS to Horizontal and Vertical components:
    Math for 3D:
        dLOS =   dE * sin(inc_angle) * sin(az_angle) * -1
               + dN * sin(inc_angle) * cos(az_angle)
               + dU * cos(inc_angle)
    Math for 2D:
        dLOS =   dH * sin(inc_angle) * cos(az_angle - az)
               + dV * cos(inc_angle)
        with dH_perp = 0.0
    This could be easily modified to support multiple view geometry
        (e.g. two adjacent tracks from asc & desc) to resolve 3D

    Parameters: los_inc_angle - 1D np.ndarray in size of (num_file), LOS incidence angle in degree.
                los_az_angle  - 1D np.ndarray in size of (num_file), LOS azimuth   angle in degree.
                horz_az_angle - float, azimuth angle for the horizontal direction of interest in degree.
                                Measured from the north with anti-clockwise direction as positive.
    Returns:    G             - 2D matrix in size of (num_file, 2)
    """
    num_file = los_inc_angle.shape[0]
    G = np.zeros((num_file, 2), dtype=np.float32)
    for i in range(num_file):
        G[i, 0] = np.sin(np.deg2rad(los_inc_angle[i])) * np.cos(np.deg2rad(los_az_angle[i] - horz_az_angle))
        G[i, 1] = np.cos(np.deg2rad(los_inc_angle[i]))

    # mute G entries for zero incidence angle (no data there, zero is the default missing pixel)
    G[los_inc_angle==0, :] = 0
    return G


def asc_desc2horz_vert(dlos, los_inc_angle, los_az_angle, horz_az_angle=-90, dlosStd=None, step=20):
    """Decompose asc / desc LOS data into horz / vert data.
    Parameters: dlos          - 3D np.ndarray in size of (num_file, length, width), LOS displacement in meters.
                los_inc_angle - 1/3D np.ndarray in size of (num_file), length, width), LOS incidence angle in degree.
                los_az_angle  - 1/3D np.ndarray in size of (num_file), length, width), LOS azimuth   angle in degree.
                horz_az_angle - float, horizontal azimuth angle of interest in degree.; 'up' assume no vertical
                step          - int, geometry step size
    Returns:    dhorz         - 2D np.ndarray in size of (length, width), horizontal displacement in meters.
                dvert         - 2D np.ndarray in size of (length, width), vertical   displacement in meters.
    """
    # initiate output
    (num_file, length, width) = dlos.shape
    num_tracks = np.nansum(dlos.astype(bool), axis=0).astype(int)

    # padded zeros in dlosStd means no data, high uncertainty
    if dlosStd is not None:
        dlosStd[dlosStd==0] = 1e9

    dhorz = np.zeros((length, width), dtype=np.float32) * np.nan
    dvert = np.zeros((length, width), dtype=np.float32) * np.nan

    # 0D (constant) incidence / azimuth angle --> invert once for all pixels
    if los_inc_angle.ndim == 1:
        if horz_az_angle == 'up':
            G = get_design_matrix4no_vert(los_inc_angle, los_az_angle)
        else:
            G = get_design_matrix4horz_vert(los_inc_angle, los_az_angle, horz_az_angle)
        if dlosStd is not None:
            # weighted LSQR
            G    *= 1 / dlosStd.reshape(num_file, -1)
            dlos *= 1 / dlosStd.reshape(num_file, -1)
        print('decomposing asc/desc into horz/vert direction ...')
        dhv = np.dot(np.linalg.pinv(G), dlos.reshape(num_file, -1)).astype(np.float32)
        dhorz = dhv[0, :].reshape(length, width)
        dvert = dhv[1, :].reshape(length, width)

    # 2D incidence / azimuth angle --> invert [window-by-window to speed up]
    elif los_inc_angle.ndim == 3:
        num_row = np.ceil(length / step).astype(int)
        num_col = np.ceil(width / step).astype(int)

        print(f'decomposing asc/desc into horz/vert direction in windows of {step}x{step} ...')
        prog_bar = ptime.progressBar(maxValue=num_row)
        for i in range(num_row):
            y0, y1 = step * i, min(step * (i + 1), length)
            for j in range(num_col):
                x0, x1 = step * j, min(step * (j + 1), width)

                # calculate the median geometry for the local window
                los_inc_angle[los_inc_angle==0] = np.nan
                los_az_angle[los_az_angle==0]   = np.nan
                med_los_inc_angle = np.nanmedian(los_inc_angle[:, y0:y1, x0:x1], axis=(1,2))
                med_los_az_angle  = np.nanmedian( los_az_angle[:, y0:y1, x0:x1], axis=(1,2))
                if dlosStd is not None:
                    med_std       = np.nanmedian(      dlosStd[:, y0:y1, x0:x1], axis=(1,2))

                if np.all(~np.isnan(med_los_inc_angle)):
                    if horz_az_angle == 'up':
                        G = get_design_matrix4no_vert(med_los_inc_angle, med_los_az_angle)
                    else:
                        G = get_design_matrix4horz_vert(med_los_inc_angle, med_los_az_angle, horz_az_angle)
                    if dlosStd is not None:
                        # weighted LSQR
                        G = np.matmul(np.diag(1/med_std), G)
                        dlos[:, y0:y1, x0:x1] = np.multiply(1/dlosStd[:, y0:y1, x0:x1], dlos[:, y0:y1, x0:x1])
                    dhv = np.dot(np.linalg.pinv(G), dlos[:, y0:y1, x0:x1].reshape(num_file, -1))
                    dhorz[y0:y1, x0:x1] = dhv[0].reshape(y1-y0, x1-x0)
                    dvert[y0:y1, x0:x1] = dhv[1].reshape(y1-y0, x1-x0)

            prog_bar.update(i+1, suffix=f'{i+1}/{num_row}')
        prog_bar.close()

    else:
        raise ValueError(f'un-supported incidence angle matrix dimension ({los_inc_angle.ndim})!')

    # mute pixels with less than 2 tracks/geometries
    dhorz[num_tracks<2] = np.nan
    dvert[num_tracks<2] = np.nan

    return dhorz, dvert


def run_asc_desc2horz_vert(inps):
    """Decompose asc / desc LOS files into horz / vert file(s).
    Parameters: inps         - namespace, input parameters
    Returns:    inps.outfile - str(s) output file(s)
    """

    ## 1. calculate the overlapping area in lat/lon
    atr_list = [readfile.read_attribute(fname, datasetName=inps.ds_name) for fname in inps.file]
    #S, N, W, E = get_overlap_lalo(atr_list)
    S, N, W, E = get_aoi_lalo(atr_list)
    lat_step = float(atr_list[0]['Y_STEP'])
    lon_step = float(atr_list[0]['X_STEP'])
    length = int(round((S - N) / lat_step))
    width  = int(round((E - W) / lon_step))
    print(f'overlaping area in SNWE: {(S, N, W, E)}')


    ## 2. read LOS data and geometry
    num_file = len(inps.file)
    dlos = np.zeros((num_file, length, width), dtype=np.float32)
    if inps.geom_file:
        los_inc_angle = np.zeros((num_file, length, width), dtype=np.float32)
        los_az_angle  = np.zeros((num_file, length, width), dtype=np.float32)
    else:
        los_inc_angle = np.zeros(num_file, dtype=np.float32)
        los_az_angle  = np.zeros(num_file, dtype=np.float32)

    for i, (atr, fname) in enumerate(zip(atr_list, inps.file)):
        # overlap SNWE --> box to read for each specific file
        coord = ut.coordinate(atr)
        y0, x0 = coord.lalo2yx(N, W)
        box = (x0, y0, x0 + width, y0 + length)

        Si, Ni, Wi, Ei, width_i, length_i = get_corners(atr)

        # initial box location (when indata perfectly overlay with outdata)
        x_start, y_start = 0, 0
        x_end = x_start + width_i
        y_end = y_start + length_i

        # update box
        if box[0]<0:
            x_start += -box[0]
            box[0] = 0
        if box[1]<0:
            y_start += -box[1]
            box[1] = 0
        if box[2]>width_i:
            box[2] = width_i
        if box[3]>length_i:
            box[3] = length_i
        box = tuple(box)
        x_end = x_start + (box[2]-box[0])
        y_end = y_start + (box[3]-box[1])
        print(f':: track {atr["trackNumber"]}; read box={box}')

        # box location
        box_loc = (x_start, y_start, x_end, y_end)

        # read data
        dlos[i, y_start:y_end, x_start:x_end] = readfile.read(fname, box=box, datasetName=inps.ds_name, no_data_values=[np.nan, 0])[0]
        msg = f'{inps.ds_name} ' if inps.ds_name else ''
        print(f'read {msg} from file: {fname}')

        # read data std if needed
        if inps.w_std:
            dlosStd = np.zeros((num_file, length, width), dtype=np.float32)
            dlosStd[i, y_start:y_end, x_start:x_end] = readfile.read(fname, box=box, datasetName=inps.ds_name+'Std', no_data_values=[np.nan, 0])[0]
            msg = f'{inps.ds_name+"Std"} ' if inps.ds_name+"Std" else ''
            print(f'read {msg} from file: {fname}')
        else:
            dlosStd = None

        # read geometry
        if inps.geom_file:
            los_inc_angle[i, y_start:y_end, x_start:x_end] = readfile.read(inps.geom_file[i], box=box, datasetName='incidenceAngle', no_data_values=[np.nan, 0])[0]
            los_az_angle[i, y_start:y_end, x_start:x_end]  = readfile.read(inps.geom_file[i], box=box, datasetName='azimuthAngle', no_data_values=[np.nan, 0])[0]
            print(f'read 2D LOS incidence / azimuth angles from file: {inps.geom_file[i]}')
        else:
            los_inc_angle[i] = ut.incidence_angle(atr, dimension=0, print_msg=False)
            los_az_angle[i] = ut.heading2azimuth_angle(float(atr['HEADING']))
            print('calculate the constant LOS incidence / azimuth angles from metadata as:')
            print(f'LOS incidence angle: {los_inc_angle[i]:.1f} deg')
            print(f'LOS azimuth   angle: {los_az_angle[i]:.1f} deg')


    ## 3. decompose LOS displacements into horizontal / Vertical displacements
    print('---------------------')
    dhorz, dvert = asc_desc2horz_vert(dlos, los_inc_angle, los_az_angle, inps.horz_az_angle, dlosStd=None)


    ## 4. write outputs
    print('---------------------')
    # Update attributes
    atr = atr_list[0].copy()
    if inps.ds_name and atr['FILE_TYPE'] in ['ifgramStack', 'timeseries', 'HDFEOS']:
        atr['FILE_TYPE'] = 'displacement'

    atr['WIDTH']  = str(width)
    atr['LENGTH'] = str(length)
    atr['X_STEP'] = str(lon_step)
    atr['Y_STEP'] = str(lat_step)
    atr['X_FIRST'] = str(W)
    atr['Y_FIRST'] = str(N)

    # update REF_X/Y
    if any(key is None for key in inps.ref_lalo):
        ref_lat, ref_lon = float(atr['REF_LAT']), float(atr['REF_LON'])
    else:
        print(':: Use your manually specified reference point')
        ref_lat, ref_lon = float(inps.ref_lalo[0]), float(inps.ref_lalo[1])
        atr['REF_LAT'] = str(ref_lat)
        atr['REF_LON'] = str(ref_lon)
    [ref_y, ref_x] = ut.coordinate(atr).geo2radar(ref_lat, ref_lon)[0:2]
    atr['REF_Y'] = int(ref_y)
    atr['REF_X'] = int(ref_x)

    # use ref_file for time-series file writing
    ref_file = inps.file[0] if atr_list[0]['FILE_TYPE'] == 'timeseries' else None

    if inps.one_outfile:
        print(f'write asc/desc/horz/vert datasets into {inps.one_outfile}')
        dsDict = {}
        for i, atr_i in enumerate(atr_list):
            # dataset name for LOS data
            track_num = atr_i.get('trackNumber', None)
            proj_name = atr_i.get('PROJECT_NAME', None)
            if proj_name in ['none', 'None', None]:
                proj_name = atr_i.get('FILE_PATH', None)
            proj_name = sensor.project_name2sensor_name(proj_name)[0]

            ds_name = proj_name if proj_name else ''
            ds_name += 'A' if atr_i['ORBIT_DIRECTION'].lower().startswith('asc') else 'D'
            ds_name += f'T{track_num}' if track_num else ''
            ds_name += '_{}'.format(atr_i['DATE12'])

            # assign dataset value
            dsDict[ds_name] = dlos[i]
        dsDict['horizontal'] = dhorz
        dsDict['vertical'] = dvert
        writefile.write(dsDict, out_file=inps.one_outfile, metadata=atr, ref_file=ref_file)

    else:
        print('writing horizontal component to file: '+inps.outfile[0])
        writefile.write(dhorz, out_file=inps.outfile[0], metadata=atr, ref_file=ref_file)
        print('writing vertical   component to file: '+inps.outfile[1])
        writefile.write(dvert, out_file=inps.outfile[1], metadata=atr, ref_file=ref_file)
        print('writing line-of-sight component to file: '+'dlos.h5')
        writefile.write(dlos, out_file='dlos.h5', metadata=atr, ref_file=ref_file)
        print('writing line-of-sight Std component to file: '+'dlosStd.h5')
        writefile.write(dlosStd, out_file='dlosStd.h5', metadata=atr, ref_file=ref_file)
        print('writing los_inc_angle to file: '+'los_inc_angle.h5')
        writefile.write(los_inc_angle, out_file='los_inc_angle.h5', metadata=atr, ref_file=ref_file)
        print('writing los_az_angle to file: '+'los_az_angle.h5')
        writefile.write(los_az_angle, out_file='los_az_angle.h5', metadata=atr, ref_file=ref_file)
    return inps.outfile
