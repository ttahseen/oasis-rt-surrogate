"""
This file contains code adapted from OASIS C code.
"""

import os
import h5py
import numpy as np

from vars import *
from model_const import *

#############################################################################################

nv = nvconstant
nvi = nv + 1

#############################################################################################


def set_extra_opacity(
    # tracers_h,
    kwave_h,
    # kopac_extra_h,
    r_cloud_h,
    Altitude_h,
    Altitudeh_h,
    Gravit,
    kpress_h,
    npress,
    nbin,
    ntra,
    nv,
):
    ccmop = np.zeros(nv + 1)
    itr = 0

    kopac_extra_h = np.zeros(shape=(nbin))
    tracers_h = np.zeros(shape=(nv * ntra + itr))

    kopac_extra_h[:nbin] = 0.0

    kopac_extra_h[10] = 0.001377  # 0.30 microns
    kopac_extra_h[11] = 0.307718  # 0.32 microns
    kopac_extra_h[12] = 0.419639  # 0.34 microns
    kopac_extra_h[13] = 0.402046  # 0.36 microns
    kopac_extra_h[14] = 0.330256  # 0.38 microns
    kopac_extra_h[15] = 0.192352  # 0.40 microns
    kopac_extra_h[16] = 0.112298  # 0.42 microns
    kopac_extra_h[17] = 0.072781  # 0.44 microns
    kopac_extra_h[18] = 0.056019  # 0.46 microns
    kopac_extra_h[19] = 0.045593  # 0.48 microns
    kopac_extra_h[20] = 0.038163  # 0.50 microns
    kopac_extra_h[21] = 0.032879  # 0.52 microns
    kopac_extra_h[22] = 0.028848  # 0.54 microns
    kopac_extra_h[23] = 0.025051  # 0.56 microns
    kopac_extra_h[24] = 0.023161  # 0.58 microns
    kopac_extra_h[25] = 0.021270  # 0.60 microns
    kopac_extra_h[26] = 0.019605  # 0.62 microns
    kopac_extra_h[27] = 0.017954  # 0.64 microns
    kopac_extra_h[28] = 0.016304  # 0.66 microns
    kopac_extra_h[29] = 0.014959  # 0.68 microns
    kopac_extra_h[30] = 0.013739  # 0.70 microns
    kopac_extra_h[31] = 0.012519  # 0.72 microns
    kopac_extra_h[32] = 0.011245  # 0.74 microns
    kopac_extra_h[33] = 0.009934  # 0.76 microns
    kopac_extra_h[34] = 0.008611  # 0.78 microns
    kopac_extra_h[35] = 0.006710  # 0.80 microns
    kopac_extra_h[36] = 0.005166  # 0.82 microns
    kopac_extra_h[37] = 0.003758  # 0.84 microns
    kopac_extra_h[38] = 0.002623  # 0.86 microns
    kopac_extra_h[39] = 0.001331  # 0.88 microns

    for lev in range(nv + 1):
        x1 = Altitudeh_h[lev] / 1000
        if x1 > 80.0:
            ccmop[lev] = 0.0
        elif x1 < 55.05:
            ccmop[lev] = 1.0
        elif x1 >= 55.05 and x1 <= 60.85:
            ccmop[lev] = 0.932 + ((0.932 - 1.000) / (60.85 - 55.05)) * (x1 - 55.05)
        elif x1 >= 60.85 and x1 <= 67.44:
            ccmop[lev] = 0.04356 + ((0.04356 - 0.932) / (67.44 - 60.85)) * (x1 - 67.44)
        else:
            ccmop[lev] = 0.0 + ((0 - 0.04356) / (80.0 - 67.44)) * (x1 - 80.0)

    for lev in range(nv):
        tracers_h[lev * ntra + itr] = ccmop[lev] - ccmop[lev + 1]

    return kopac_extra_h, tracers_h, ccmop


def genesis_read_input_files(
    fnr,
    datapath=DATAPATH,
):
    # Open the first HDF5 file
    with h5py.File(
        os.path.join(datapath, f"oasis_output_Venus_{fnr}.h5"),
        "r"
        # f"/Users/ttahseen/Documents/research/venus-simulation/data/output/oasis_output_Venus_{fnr}.h5", "r"
    ) as file_id:
        # Read the datasets from the first file
        Rho_3D_h = file_id["Rho"][:]
        Pressure_3D_h = file_id["Pressure"][:]
        Pressureh_3D_h = file_id["Pressureh"][:]
        Temperature_3D_h = file_id["Temperature"][:]

        sTemperature_3D_h = file_id["sTemperature"][:]
        alb_surf_lw_3D = file_id["alb_surf_lw"][:]
        alb_surf_sw_3D = file_id["alb_surf_sw"][:]
        cosz_3D = file_id["cosz"][:]

        fnet_dn_lw_3D_h = file_id["fnet_dn_lw_h"][:]
        fnet_dn_sw_3D_h = file_id["fnet_dn_sw_h"][:]
        fnet_up_lw_3D_h = file_id["fnet_up_lw_h"][:]
        fnet_up_sw_3D_h = file_id["fnet_up_sw_h"][:]

        # Rho_3D_h = np.reshape(file_id["Rho"][:], newshape=(numhz, nv))
        # Pressure_3D_h = np.reshape(file_id["Pressure"][:], newshape=(numhz, nv))
        # Pressureh_3D_h = np.reshape(file_id["Pressureh"][:], newshape=(numhz, nvi))
        # Temperature_3D_h = np.reshape(file_id["Temperature"][:], newshape=(numhz, nv))

        # sTemperature_3D_h = np.reshape(file_id["sTemperature"][:], newshape=(numhz))
        # alb_surf_lw_3D = np.reshape(file_id["alb_surf_lw"][:], newshape=(numhz))
        # alb_surf_sw_3D = np.reshape(file_id["alb_surf_sw"][:], newshape=(numhz))
        # cosz_3D = np.reshape(file_id["cosz"][:], newshape=(numhz))

        # fnet_dn_lw_3D_h = np.reshape(file_id["fnet_dn_lw_h"][:], newshape=(numhz, nvi))
        # fnet_dn_sw_3D_h = np.reshape(file_id["fnet_dn_sw_h"][:], newshape=(numhz, nvi))
        # fnet_up_lw_3D_h = np.reshape(file_id["fnet_up_lw_h"][:], newshape=(numhz, nvi))
        # fnet_up_sw_3D_h = np.reshape(file_id["fnet_up_sw_h"][:], newshape=(numhz, nvi))
    return (
        Rho_3D_h,
        Pressure_3D_h,
        Pressureh_3D_h,
        Temperature_3D_h,
        sTemperature_3D_h,
        alb_surf_lw_3D,
        alb_surf_sw_3D,
        cosz_3D,
        fnet_dn_lw_3D_h,
        fnet_dn_sw_3D_h,
        fnet_up_lw_3D_h,
        fnet_up_sw_3D_h,
    )


def genesis_read_output_grid(datapath=DATAPATH):
    with h5py.File(
        os.path.join(datapath, "oasis_output_grid_Venus.h5"),
        "r"
        # "/Users/ttahseen/Documents/research/venus-simulation/data/all_data/tmp-data/oasis_output_grid_Venus.h5",
    ) as file_id:
        # Read the datasets from the second file
        Altitude_h = file_id["Altitude"][:]
        Altitudeh_h = file_id["Altitudeh"][:]
        lonlat_h = file_id["lonlat"][:]

        # Altitude_h = np.reshape(file_id["Altitude"][:], newshape=(nv))
        # Altitudeh_h = np.reshape(file_id["Altitudeh"][:], newshape=(nvi))
        # lonlat_h = np.reshape(file_id["lonlat"][:], newshape=(numhz, 2))
    return Altitude_h, Altitudeh_h, lonlat_h


def genesis_read_params_file(datapath=INPUT_PATH):
    with h5py.File(
        os.path.join(datapath, "opacities_venus.h5"),
        "r"
        # "/Users/ttahseen/Documents/research/venus-simulation/data/input/opacities_venus.h5", "r"
    ) as file_id:
        # Gas absorption
        dg_h = file_id["DELG"][:]
        kpress_h = file_id["PK"][:]
        ktemp_h = file_id["TK"][:]
        kopac_h = file_id["gasabs"][:]
        kwave_max_h = file_id["wavmax"][:]
        kwave_h = file_id["wavmid"][:]
        kwave_min_h = file_id["wavmin"][:]

        # Clouds
        kopac_ext_h = file_id["kopac_ext"][:]
        kopac_f0_h = file_id["kopac_f0"][:]
        kopac_g0_h = file_id["kopac_g0"][:]
        kopac_w0_h = file_id["kopac_w0"][:]
        pcld_h = file_id["pcld"][:]
        r_cloud_h = file_id["r_cloud"][:]

        # Rayleigh Scattering
        rscat_h = file_id["rscattering"][:]

        # # Gas absorption
        # dg_h = np.reshape(file_id["DELG"], newshape=(ny))
        # kpress_h = np.reshape(file_id["PK"], newshape=(npress))
        # ktemp_h = np.reshape(file_id["TK"], newshape=(ntemp))
        # kopac_h = np.reshape(file_id["gasabs"], newshape=(ny, nbin, ntemp, npress))
        # kwave_max_h = np.reshape(file_id["wavmax"], newshape=(nbin))
        # kwave_h = np.reshape(file_id["wavmid"], newshape=(nbin))
        # kwave_min_h = np.reshape(file_id["wavmin"], newshape=(nbin))

        # # Clouds
        # kopac_ext_h = np.reshape(file_id["kopac_ext"], newshape=(nbin, npcloud))
        # kopac_f0_h = np.reshape(file_id["kopac_f0"], newshape=(nbin, npcloud))
        # kopac_g0_h = np.reshape(file_id["kopac_g0"], newshape=(nbin, npcloud))
        # kopac_w0_h = np.reshape(file_id["kopac_w0"], newshape=(nbin, npcloud))
        # pcld_h = np.reshape(file_id["pcld"], newshape=(npcloud))
        # r_cloud_h = np.reshape(file_id["r_cloud"], newshape=(npcloud))

        # # Rayleigh Scattering
        # rscat_h = np.reshape(file_id["rscattering"], newshape=(nbin, ntemp, npress))

        # md = file_id["md"]

    with h5py.File(
        # "/Users/ttahseen/Documents/research/venus-simulation/data/input/stars.h5", "r"
        os.path.join(datapath, "stars.h5"),
        "r",
    ) as file_id:
        # StarF_h = np.reshape(file_id["sunFlx"], newshape=(nbin))
        StarF_h = file_id["sunFlx"][:]
        # hatp7 = file_id["hatp7"]

    return (
        dg_h,
        kpress_h,
        ktemp_h,
        kopac_h,
        kwave_max_h,
        kwave_h,
        kwave_min_h,
        kopac_ext_h,
        kopac_f0_h,
        kopac_g0_h,
        kopac_w0_h,
        pcld_h,
        r_cloud_h,
        rscat_h,
        StarF_h,
    )


def calc_deltcol(
    Temperature_d, Pressure_d, Pressureh_d, Rho_d, Altitude_d, Altitudeh_d, Gravit, npress, ntemp
):
    deltacolumn_d = np.zeros(shape=(nv))
    for lev in range(nv - 1):
        ptop = Pressureh_d[lev] * 10.0
        pbot = Pressureh_d[lev + 1] * 10.0

        deltacolumn_d[lev] = (pbot - ptop) / (Gravit * 100)
        if lev == nv - 1:
            deltacolumn_d[nv] = ptop / (Gravit * 100)

    return deltacolumn_d


# def calc_deltcol(
#     Temperature_d, Pressure_d, Pressureh_d, Rho_d, Altitude_d, Altitudeh_d, Gravit, npress, ntemp
# ):
#     deltacolumn_d = np.zeros(shape=(nv + 1))
#     for lev in range(nv + 1):
#         ptop = Pressureh_d[lev] * 10.0
#         pbot = Pressureh_d[lev + 1] * 10.0

#         deltacolumn_d[lev] = (pbot - ptop) / (Gravit * 100)
#         if lev == nv:
#             deltacolumn_d[nv] = ptop / (Gravit * 100)

#     return deltacolumn_d
