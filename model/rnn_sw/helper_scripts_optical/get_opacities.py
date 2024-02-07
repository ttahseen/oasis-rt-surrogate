"""
This file contains code to calculate and save opacity variables across layers in an
atmospheric column.

This code has been adapted from OASIS C code (https://github.com/jmmendonca/oasis-rt/tree/main).
"""

from tqdm import tqdm
from model_const import *
from cyclops_rad_sw import *
from genesis_input_files import *

from vars import *

#############################################################################################

# Loading data


def generate_opacities_arrays(datapath, input_path):
    (
        Altitude_h,
        Altitudeh_h,
        lonlat_h,
    ) = genesis_read_output_grid(datapath)

    (
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
    ) = genesis_read_params_file(input_path)

    # for fnr in tqdm(range(1, 20)):
    frange = [
        int(f.replace(".h5", "").lstrip("oasis_output_Venus_"))
        for f in os.listdir(os.path.join(datapath))
        if "oasis_output_Venus_" in f
    ]
    for fnr in frange:
        (
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
        ) = genesis_read_input_files(fnr=fnr, datapath=datapath)

        # Setting extra opacity

        kopac_extra_h, tracers_h, ccmop = set_extra_opacity(
            kwave_h,
            r_cloud_h,
            Altitude_h,
            Altitudeh_h,
            Gravit,
            kpress_h,
            npress,
            nbin,
            ntra,
            nv,
        )

        # Getting opacities across columns

        (
            Rho_d,
            Pressure_d,
            Pressureh_d,
            Temperature_d,
            sTemperature_d,
            alb_surf_lw_3D,
            alb_surf_sw_3D,
            cosz_d,
            fnet_dn_lw_d,
            fnet_dn_sw_d,
            fnet_up_lw_d,
            fnet_up_sw_d,
            Altitude_d,
            Altitudeh_d,
            lonlat_d,
        ) = (
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
            Altitude_h,
            Altitudeh_h,
            lonlat_h,
        )

        ktemp_d = ktemp_h
        kpress_d = kpress_h
        kopac_d = kopac_h
        kopac_extra_d = kopac_extra_h
        rscat_d = rscat_h
        kopac_ext_d = kopac_ext_h
        pcld_d = pcld_h
        kopac_w0_d = kopac_w0_h
        kopac_g0_d = kopac_g0_h
        kopac_f0_d = kopac_f0_h
        tracers_d = tracers_h
        deltacolumn_d = calc_deltcol(
            Temperature_d, Pressure_d, Pressureh_d, Rho_d, Altitude_d, Altitudeh_d, Gravit, npress, ntemp
        )

        opac, opac_ray, opac_ext = fnet_rad_sw_sca(
            cosz_d,
            ktemp_d,
            kpress_d,
            kopac_d,
            kopac_extra_d,
            rscat_d,
            kopac_ext_d,
            pcld_d,
            kopac_w0_d,
            kopac_g0_d,
            kopac_f0_d,
            deltacolumn_d,
            Temperature_d,
            Pressure_d,
            Pressureh_d,
            tracers_d,
            Rho_d,
            Altitude_d,
            Altitudeh_d,
        )

        # Saving files
        opacities_dir = os.path.join(datapath, "rnn_sw", "optical", "optical_variables")
        if not os.path.isdir(opacities_dir):
            os.mkdir(opacities_dir)

        with open(os.path.join(opacities_dir, f"opac-{fnr}.npy"), "wb") as f:
            np.save(f, opac)
        with open(os.path.join(opacities_dir, f"opac_ray-{fnr}.npy"), "wb") as f:
            np.save(f, opac_ray)
        with open(os.path.join(opacities_dir, f"opac_ext-{fnr}.npy"), "wb") as f:
            np.save(f, opac_ext)
        with open(os.path.join(opacities_dir, f"cosz_d-{fnr}.npy"), "wb") as f:
            np.save(f, cosz_d)
        with open(os.path.join(opacities_dir, f"alb_surf_sw_3D-{fnr}.npy"), "wb") as f:
            np.save(f, alb_surf_sw_3D)
