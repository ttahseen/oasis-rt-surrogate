"""
This file contains code adapted from OASIS C code.
"""

import math
import numpy as np
from tqdm import tqdm
from numba import cuda
from model_const import *

#############################################################################################

# SCATTERING


def taus_f(w, f, t):
    """Equation 8"""
    return (1.0 - w * f) * t


def omgs_f(w, f):
    """Equation 9"""
    return (1.0 - f) * w / (1.0 - w * f)


def asys_f(g, f):
    """Equation 10"""
    return (g - f) / (1.0 - f)


def alpha_f(w, uu, g, e):
    """Equation 15"""
    return 0.75 * w * uu * ((1.0 + g * (1.0 - w)) / (1.0 - e * e * uu * uu))


def gamma_f(w, uu, g, e):
    """Equation 16"""
    return 0.50 * w * ((3.0 * g * (1.0 - w) * uu * uu + 1.0) / (1.0 - e * e * uu * uu))


def n_f(uu, et):
    """Equation 17"""
    return ((uu + 1.0) * (uu + 1.0) / et) - ((uu - 1.0) * (uu - 1.0) * et)


def u_f(w, g, e):
    """Equation 18"""
    return 1.5 * (1.0 - w * g) / e


def el_f(w, g):
    """Equation 19"""
    return math.sqrt(3.0 * (1 - w) * (1.0 - w * g))


#############################################################################################


def rad_ed(
    coszi,
    Altitude_d,
    A,
    rdir,
    rdif,
    tdir,
    tdif,
    explay,
    exptdn,
    rdndif,
    tottrn,
    opac,
    opac_ray,
    opac_ext,
    w0,
    g0,
    f0,
    nv,
):
    """
    The solar radiation code is based on a two-stream solution that incorporates a δ-Eddington approximation and a layer-adding method.
    """
    valmax = 0.999999
    argmax = 99.0

    ##### DELTA-EDDINGTON COMPUTATION ####
    # Start at top and go down one layer at a time
    for lev in range(nv, 0, -1):
        # Calculates the solar beam transmission, total transmission, and reflectivity for diffuse radiation

        ##### Solar beam transmission
        cosz = math.sqrt(1.0 - pow(A / (A + Altitude_d[lev - 1]), 2.0) * (1.0 - coszi * coszi))

        if lev != nv:  # If level is not the top level
            # Opacity
            exptdn[lev] = exptdn[lev + 1] * explay[lev + 1]
            rdenom = 1.0 / (1.0 - rdif[lev + 1] * rdndif[lev + 1])
            rdirexp = rdir[lev + 1] * exptdn[lev + 1]
            tdnmexp = tottrn[lev + 1] - exptdn[lev + 1]
            ##### T_12 Total transmission of direct radiation between two layers(Equation 22)
            tottrn[lev] = (
                exptdn[lev + 1] * tdir[lev + 1]
                + tdif[lev + 1] * (tdnmexp + rdndif[lev + 1] * rdirexp) * rdenom
            )
            ##### Total reflection of diffuse radiation between two layers
            rdndif[lev] = rdif[lev + 1] + (rdndif[lev + 1] * tdif[lev + 1]) * (tdif[lev + 1] * rdenom)
        else:  # For top level
            exptdn[nv] = 1.0
            rdndif[nv] = 0.0
            tottrn[nv] = 1.0

            rdir[nv] = 0.0
            rdif[nv] = 0.0
            tdir[nv] = 1.0
            tdif[nv] = 1.0
            explay[nv] = 1.0

        # Compute next layer delta-Eddington solution only if total transmission of radiation to the interface
        # just above the layer exceeds t rmin
        tautot = opac_ray[lev - 1] + opac_ext[lev - 1] + opac[lev - 1]
        wtau = opac_ray[lev - 1] + w0[lev - 1]

        if wtau != 0:
            # Equation 5
            wtot = wtau / tautot
            # Equation 6
            gtot = (g0[lev]) / wtau
            # Equation 7
            ftot = ((opac_ray[lev] * 0.5) / wtau) + ((f0[lev]) / wtau)
        else:
            wtot = 0.0
            gtot = 0.0
            ftot = 0.0

        if wtot > valmax:
            wtot = valmax
        if gtot > valmax:
            gtot = valmax
        if ftot > valmax:
            ftot = valmax

        if wtot < 0.0:
            wtot = 0.0
        if gtot < 0.0:
            gtot = 0.0
        if ftot < 0.0:
            ftot = 0.0

        ts = taus_f(wtot, ftot, tautot)
        ws = omgs_f(wtot, ftot)
        gs = asys_f(gtot, ftot)
        lm = el_f(ws, gs)
        alp = alpha_f(ws, cosz, gs, lm)
        gam = gamma_f(ws, cosz, gs, lm)
        ue = u_f(ws, gs, lm)

        arg = lm * ts

        if arg > argmax:
            arg = argmax

        extins = math.exp(-arg)
        ne = n_f(ue, extins)
        # Diffuse Reflectivity (Equation 13)
        rdif[lev] = (ue + 1.0) * (ue - 1.0) * (1.0 / extins - extins) / ne
        # Diffuse Transmissivity (Equation 14)
        tdif[lev] = 4.0 * ue / ne

        arg = ts / cosz

        if arg > argmax:
            arg = argmax

        explay[lev] = math.exp(-arg)
        apg = alp + gam
        amg = alp - gam

        # Direct Reflectivity (Equation 11)
        rdir[lev] = amg * (tdif[lev] * explay[lev] - 1.0) + apg * rdif[lev]
        # Direct Transmissivity (Equation 12)
        tdir[lev] = apg * tdif[lev] + (amg * rdif[lev] - (apg - 1.0)) * explay[lev]

        if rdir[lev] < 0.0:
            rdir[lev] = 0.0
        if tdir[lev] < 0.0:
            tdir[lev] = 0.0
        if rdif[lev] < 0.0:
            rdif[lev] = 0.0
        if tdif[lev] < 0.0:
            tdif[lev] = 0.0

    lev = 0

    exptdn[lev] = exptdn[lev + 1] * explay[lev + 1]
    rdenom = 1.0 / (1.0 - rdif[lev + 1] * rdndif[lev + 1])
    rdirexp = rdir[lev + 1] * exptdn[lev + 1]
    tdnmexp = tottrn[lev + 1] - exptdn[lev + 1]
    tottrn[lev] = (
        exptdn[lev + 1] * tdir[lev + 1] + tdif[lev + 1] * (tdnmexp + rdndif[lev + 1] * rdirexp) * rdenom
    )
    rdndif[lev] = rdif[lev + 1] + (rdndif[lev + 1] * tdif[lev + 1]) * (tdif[lev + 1] * rdenom)


def static_cloud_opacity_sw(
    x,
    w0,
    g0,
    f0,
    ptop,
    pbot,
    pcld_d,
    deltaopacpress_cld,
    kopac_ext_d,
    kopac_w0_d,
    kopac_g0_d,
    kopac_f0_d,
    npcloud,
):
    p = (math.log10(pbot) - math.log10(pcld_d[0])) / deltaopacpress_cld
    if p > npcloud - 1:
        p = npcloud - 1.0001
    if p < 0:
        p = 0.0001
    pdown = math.floor(p)
    pup = math.ceil(p)
    value = 0.0

    if pdown != pup:
        value = kopac_ext_d[x * npcloud + pdown] * (pup - p) + kopac_ext_d[x * npcloud + pup] * (p - pdown)
        w0 = kopac_w0_d[x * npcloud + pdown] * (pup - p) + kopac_w0_d[x * npcloud + pup] * (p - pdown)
        g0 = kopac_g0_d[x * npcloud + pdown] * (pup - p) + kopac_g0_d[x * npcloud + pup] * (p - pdown)
        f0 = kopac_f0_d[x * npcloud + pdown] * (pup - p) + kopac_f0_d[x * npcloud + pup] * (p - pdown)
    elif pdown == pup:
        value = kopac_ext_d[x * npcloud + pdown]
        w0 = kopac_w0_d[x * npcloud + pdown]
        g0 = kopac_g0_d[x * npcloud + pdown]
        f0 = kopac_f0_d[x * npcloud + pdown]

    p = (math.log10(ptop) - math.log10(pcld_d[0])) / deltaopacpress_cld
    if p > npcloud - 1:
        p = npcloud - 1.0001
    if p < 0:
        p = 0.0001
    pdown = math.floor(p)
    pup = math.ceil(p)

    if pdown != pup:
        value = value - (
            kopac_ext_d[x * npcloud + pdown] * (pup - p) + kopac_ext_d[x * npcloud + pup] * (p - pdown)
        )
        w0 = w0 - (kopac_w0_d[x * npcloud + pdown] * (pup - p) + kopac_w0_d[x * npcloud + pup] * (p - pdown))
        g0 = g0 - (kopac_g0_d[x * npcloud + pdown] * (pup - p) + kopac_g0_d[x * npcloud + pup] * (p - pdown))
        f0 = f0 - (kopac_f0_d[x * npcloud + pdown] * (pup - p) + kopac_f0_d[x * npcloud + pup] * (p - pdown))
    elif pdown == pup:
        value = value - kopac_ext_d[x * npcloud + pdown]
        w0 = w0 - kopac_w0_d[x * npcloud + pdown]
        g0 = g0 - kopac_g0_d[x * npcloud + pdown]
        f0 = f0 - kopac_f0_d[x * npcloud + pdown]

    return value


def fnet_rad_sw_sca(
    cosz_d,
    # fnet_up_d,
    # fnet_dn_d,
    ktemp_d,
    kpress_d,
    kopac_d,
    kopac_extra_d,
    rscat_d,
    kopac_ext_d,
    # r_cloud_d,
    pcld_d,
    kopac_w0_d,
    kopac_g0_d,
    kopac_f0_d,
    # dkwave_d,
    deltacolumn_d,
    Temperature_d,
    # StarF_d,
    Pressure_d,
    Pressureh_d,
    tracers_d,
    Rho_d,
    # dg_d,
    # A,  # planet_params
    # Dist_planet_star,
    # Radius_star,
    # Surf_alb_sw,
    Altitude_d,
    Altitudeh_d,
    # Gravit,  # planet_params
    # nbinsw,
    # npress,
    # ntemp,
    # ny,
    # nbin,
    # npcloud,
    # ntra,
):
    nv = nvconstant

    # x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    # coszi = cosz_d[0]

    tdown = 0
    tup = 0
    pdown = 0
    pup = 0
    t = 0.0
    p = 0.0
    pp = 0.0
    ptop = 0.0
    pbot = 0.0

    f_up_wk = np.zeros(nv + 1)
    f_dn_wk = np.zeros(nv + 1)

    rdir = np.zeros(nv + 1)
    rdif = np.zeros(nv + 1)
    tdir = np.zeros(nv + 1)
    tdif = np.zeros(nv + 1)

    rupdir = np.zeros(nv + 1)
    rupdif = np.zeros(nv + 1)

    explay = np.zeros(nv + 1)
    exptdn = np.zeros(nv + 1)
    rdndif = np.zeros(nv + 1)
    tottrn = np.zeros(nv + 1)

    opac = np.zeros(nv + 1)
    opac_ray = np.zeros(nv + 1)

    opac_ext = np.zeros(nv + 1)
    w0 = np.zeros(nv + 1)
    g0 = np.zeros(nv + 1)
    f0 = np.zeros(nv + 1)
    rdnm = 0.0

    for x in range(nbinsw):
        for y in range(ny):
            # if (x < nbinsw) & (y < ny): # Number of shortwave spectral bins x Number of Gaussian points in the ktable == (148 x 20)
            # if coszi >= 0:
            for lev in range(nv + 1):
                opac[lev] = 0.0
                opac_ray[lev] = 0.0
                opac_ext[lev] = 0.0
                w0[lev] = 0.0
                g0[lev] = 0.0
                f0[lev] = 0.0

                f_dn_wk[lev] = 0.0
                f_up_wk[lev] = 0.0

                rdir[lev] = 0.0
                rdif[lev] = 0.0
                tdir[lev] = 0.0
                tdif[lev] = 0.0
                rupdir[lev] = 0.0
                rupdif[lev] = 0.0
                explay[lev] = 0.0
                exptdn[lev] = 0.0
                rdndif[lev] = 0.0
                tottrn[lev] = 0.0

            deltaopactemp = (ktemp_d[ntemp - 1] - ktemp_d[0]) / (ntemp - 1.0)
            deltaopacpress = (math.log10(kpress_d[npress - 1]) - math.log10(kpress_d[0])) / (npress - 1.0)
            deltaopacpress_cld = (math.log10(pcld_d[npcloud - 1]) - math.log10(pcld_d[0])) / (npcloud - 1.0)

            #   Rayleigh Scattering and gas opacity
            # for lev in range(nv + 1):
            for lev in range(nv):
                rs = 0.0
                dc = deltacolumn_d[lev]
                if lev < nv:
                    t = (Temperature_d[lev] - ktemp_d[0]) / deltaopactemp
                    if t > ntemp - 1:
                        t = ntemp - 1.0001
                    if t < 0:
                        t = 0.0001

                    tdown = np.floor(t)
                    tup = np.ceil(t)
                    p = (math.log10(Pressure_d[lev]) - math.log10(kpress_d[0])) / deltaopacpress

                    if p > npress - 1:
                        p = npress - 1.0001
                    if p < 0:
                        p = 0.0001
                    pdown = np.floor(p)
                    pup = np.ceil(p)

                else:
                    t = (Temperature_d[nv - 1] - ktemp_d[0]) / deltaopactemp
                    if t > ntemp - 1:
                        t = ntemp - 1.0001
                    if t < 0:
                        t = 0.0001

                    tdown = np.floor(t)
                    tup = np.ceil(t)

                    pp = Pressure_d[nv - 2] - Rho_d[nv - 1] * Gravit * (
                        2 * Altitudeh_d[nv] - Altitude_d[nv - 1] - Altitude_d[nv - 2]
                    )
                    if pp < 0:
                        pp = 0.0
                    ptop = 0.5 * (Pressure_d[nv - 1] + pp)

                    p = (math.log10(ptop) - math.log10(kpress_d[0])) / deltaopacpress

                    if p > npress - 1:
                        p = npress - 1.0001
                    if p < 0:
                        p = 0.0001

                    pdown = np.floor(p)
                    pup = np.ceil(p)

                #   interpolate layer and interface opacities from opacity table
                if (pdown != pup) & (tdown != tup):
                    opac[lev] = (
                        kopac_d[int(y + ny * x + ny * nbin * pdown + ny * nbin * npress * tdown)]
                        * (pup - p)
                        * (tup - t)
                        + kopac_d[int(y + ny * x + ny * nbin * pup + ny * nbin * npress * tdown)]
                        * (p - pdown)
                        * (tup - t)
                        + kopac_d[int(y + ny * x + ny * nbin * pdown + ny * nbin * npress * tup)]
                        * (pup - p)
                        * (t - tdown)
                        + kopac_d[int(y + ny * x + ny * nbin * pup + ny * nbin * npress * tup)]
                        * (p - pdown)
                        * (t - tdown)
                    ) * dc

                    if lev < nv:
                        rs = (
                            rscat_d[int(x + nbin * pdown + nbin * npress * tdown)] * (pup - p) * (tup - t)
                            + rscat_d[int(x + nbin * pup + nbin * npress * tdown)] * (p - pdown) * (tup - t)
                            + rscat_d[int(x + nbin * pdown + nbin * npress * tup)] * (pup - p) * (t - tdown)
                            + rscat_d[int(x + nbin * pup + nbin * npress * tup)] * (p - pdown) * (t - tdown)
                        )

                        opac_ray[lev] = dc * rs * (Gravit * 100) / 1e6

                elif tdown == tup & pdown != pup:
                    opac[lev] = (
                        kopac_d[int(y + ny * x + ny * nbin * pdown + ny * nbin * npress * tdown)] * (pup - p)
                        + kopac_d[int(y + ny * x + ny * nbin * pup + ny * nbin * npress * tdown)]
                        * (p - pdown)
                    ) * dc

                    if lev < nv:
                        rs = rscat_d[int(x + nbin * pdown + nbin * npress * tdown)] * (pup - p) + rscat_d[
                            x + nbin * pup + nbin * npress * tdown
                        ] * (p - pdown)

                        opac_ray[lev] = dc * rs * (Gravit * 100) / 1e6

                elif pdown == pup & tdown != tup:
                    opac[lev] = (
                        kopac_d[int(y + ny * x + ny * nbin * pdown + ny * nbin * npress * tdown)] * (tup - t)
                        + kopac_d[int(y + ny * x + ny * nbin * pdown + ny * nbin * npress * tup)]
                        * (t - tdown)
                    ) * dc

                    if lev < nv:
                        rs = rscat_d[int(x + nbin * pdown + nbin * npress * tdown)] * (tup - t) + rscat_d[
                            int(x + nbin * pdown + nbin * npress * tup)
                        ] * (t - tdown)

                        opac_ray[lev] = dc * rs * (Gravit * 100) / 1e6

                elif tdown == tup & pdown == pup:
                    opac[lev] = kopac_d[int(y + ny * x + ny * nbin * pdown + ny * nbin * npress * tdown)] * dc
                    if lev < nv:
                        rs = rscat_d[int(x + nbin * pdown + nbin * npress * tdown)]

                        opac_ray[lev] = dc * rs * (Gravit * 100) / 1e6

                if lev < nv:
                    #   Cloud opacity
                    pbot = Pressureh_d[lev]
                    ptop = Pressureh_d[lev + 1]

                    opac_ext[lev] = static_cloud_opacity_sw(
                        x,
                        w0[lev],
                        g0[lev],
                        f0[lev],
                        ptop,
                        pbot,
                        pcld_d,
                        deltaopacpress_cld,
                        kopac_ext_d,
                        kopac_w0_d,
                        kopac_g0_d,
                        kopac_f0_d,
                        npcloud,
                    )

            # Extra Absorber
            for lev in range(nv):
                opac[lev] = opac[lev] + kopac_extra_d[x] * tracers_d[lev * ntra + 0]

    cosz_binary = np.repeat(np.expand_dims((cosz_d > 0).astype("int"), axis=1), repeats=50, axis=1)

    opac_all = np.repeat(np.expand_dims(opac, axis=0), repeats=len(cosz_d), axis=0)
    opac_ray_all = np.repeat(np.expand_dims(opac_ray, axis=0), repeats=len(cosz_d), axis=0)
    opac_ext_all = np.repeat(np.expand_dims(opac_ext, axis=0), repeats=len(cosz_d), axis=0)

    opac_all = np.multiply(opac_all, cosz_binary)
    opac_ray_all = np.multiply(opac_ray_all, cosz_binary)
    opac_ext_all = np.multiply(opac_ext_all, cosz_binary)

    return opac_all, opac_ray_all, opac_ext_all


def get_all_cols(mat, cosz_d):
    cosz_binary = np.repeat(np.expand_dims((cosz_d > 0).astype("int"), axis=1), repeats=50, axis=1)
    mat_all = np.multiply(mat, cosz_binary)
    return mat_all
    # ##### AT THIS POINT, SHOULD SAVE ALL VARS AND USE AS INPUT FOR TRAINING MODEL TO EMULATE [Rad_ed(...) FUNCTION, DIR AND DIF REFLECTIVITY, UP AND DOWN FLUXES, UPWARDS AND DOWNWARDS FLUXES]

    # rad_ed(
    #     coszi,
    #     Altitude_d,
    #     A,
    #     rdir,
    #     rdif,
    #     tdir,
    #     tdif,
    #     explay,
    #     exptdn,
    #     rdndif,
    #     tottrn,
    #     opac,
    #     opac_ray,
    #     opac_ext,
    #     w0,
    #     g0,
    #     f0,
    #     nv,
    # )

    # # //
    # # //         Compute reflectivity to direct and diffuse radiation for layers.
    # # //
    # rupdir[0] = Surf_alb_sw
    # rupdif[0] = Surf_alb_sw

    # for lev in range(nv + 1):
    #     rdnm = 1.0 / (1.0 - rdif[lev] * rupdif[lev - 1])
    #     rupdir[lev] = (
    #         rdir[lev]
    #         + tdif[lev]
    #         * (rupdir[lev - 1] * explay[lev] + rupdif[lev - 1] * (tdir[lev] - explay[lev]))
    #         * rdnm
    #     )
    #     rupdif[lev] = rdif[lev] + rupdif[lev - 1] * tdif[lev] * tdif[lev] * rdnm

    # # //
    # # //          Compute up and down fluxes for each interface
    # # //
    # for lev in range(nv + 1):
    #     rdnm = 1.0 / (1.0 - rdndif[lev] * rupdif[lev])
    #     f_up_wk[lev] = (exptdn[lev] * rupdir[lev] + (tottrn[lev] - exptdn[lev]) * rupdif[lev]) * rdnm
    #     f_dn_wk[lev] = (
    #         exptdn[lev] + (tottrn[lev] - exptdn[lev] + exptdn[lev] * rupdir[lev] * rdndif[lev]) * rdnm
    #     )

    # # //          Calculates the integrated upwards and downwards fluxes
    # ftop_model = coszi * StarF_d[x] * pow(Radius_star / Dist_planet_star, 2.0)
    # dg = dg_d[y] * dkwave_d[x] * ftop_model
    # # for lev in range(nv+1):
    # #      myatomicAdd(&(fnet_up_d[lev]), f_up_wk[lev] * dg)
    # #      myatomicAdd(&(fnet_dn_d[lev]), f_dn_wk[lev] * dg)
