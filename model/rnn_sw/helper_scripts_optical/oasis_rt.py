class OASISRT:
    def __init__(self, numhz, nv, nvi, nbin, nbinsw, nbinlw, ntemp, npress, ny, npcloud, ntra):
        self.numhz = numhz
        self.nv = nv
        self.nvi = nvi
        self.nbin = nbin
        self.nbinsw = nbinsw
        self.nbinlw = nbinlw
        self.ntemp = ntemp
        self.npress = npress
        self.ny = ny
        self.npcloud = npcloud
        self.ntra = ntra

        # Host Grid
        self.Altitude_h = None
        self.Altitudeh_h = None
        self.lonlat_h = None

        # Host Atmosphere
        self.Rho_h = None
        self.Pressure_h = None
        self.Pressureh_h = None
        self.Temperature_h = None
        self.fnet_up_sw_h = None
        self.fnet_dn_sw_h = None
        self.fnet_up_lw_h = None
        self.fnet_dn_lw_h = None

        # Host Tracers
        self.tracers_h = None

        # Host Radiation
        self.kwave_h = None

        # Host Surface & Soil
        self.sTemperature_h = None
        self.cosz_h = None
        self.Pressure_3D_h = None
        self.Pressureh_3D_h = None
        self.Rho_3D_h = None
        self.Temperature_3D_h = None
        self.alb_surf_lw_3D = None
        self.alb_surf_sw_3D = None
        self.cosz_3D = None
        self.fnet_dn_lw_3D_h = None
        self.fnet_dn_sw_3D_h = None
        self.fnet_up_lw_3D_h = None
        self.fnet_up_sw_3D_h = None
        self.sTemperature_3D_h = None

        # Device Grid
        self.Altitude_d = None
        self.Altitudeh_d = None

        # Device Atmosphere
        self.Temperature_d = None
        self.Rho_d = None
        self.Pressure_d = None
        self.Pressureh_d = None

        # Device Radiative transfer
        self.planck_grid_d = None
        self.deltacolumn_d = None
        self.fnet_up_sw_d = None
        self.fnet_dn_sw_d = None
        self.fnet_up_lw_d = None
        self.fnet_dn_lw_d = None
        self.kwave_d = None
        self.kwave_int_d = None
        self.dkwave_d = None
        self.kpress_d = None
        self.ktemp_d = None
        self.kopac_d = None
        self.kopac_extra_d = None
        self.dg_d = None
        self.kopac_ext_d = None
        self.r_cloud_d = None
        self.pcld_d = None
        self.kopac_w0_d = None
        self.kopac_g0_d = None
        self.kopac_f0_d = None
        self.rscat_d = None
        self.StarF_d = None
        self.cosz_d = None

        # Device Tracers
        self.tracers_d = None

        # Device Surface & Soil
        self.sTemperature_d = None

    def AllocData(self):
        pass

    def InitialValues(
        self,
        fnr,
        inum,
        Dist_planet_star,
        Radius_star,
        Surf_alb_sw,
        Surf_alb_lw,
        A,
        Cp,
        Rd,
        P_Ref,
        Gravit,
        Mmol,
        ifile_ktable,
        ifile_star1,
        ifile_star2,
    ):
        pass

    def PROFX(self, Cp, Rd, Mmol, P_Ref, Gravit, A, Dist_planet_star, Radius_star, Surf_alb_sw, Surf_alb_lw):
        pass

    def CopyToHost(self):
        pass

    def Output(self, inum):
        pass

    def __del__(self):
        pass
