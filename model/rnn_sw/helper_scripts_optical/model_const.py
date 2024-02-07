# GRID
nvconstant = 49  # Number of vertical layers in the atmosphere
numhz = 10242  # Number of points (3D spherical surface)

# STAR
Star_ID = "Sun"
Radius_star = 695508000.0  # Star radius [m]
STeff = 5772.0  # Star effective temperature (photosphere) [K]

# PLANET
Planet_ID = "Venus"
A = 6052000.0  # Planet radius [m]
Gravit = 8.87  # Gravitational acceleration [m/s^2]
L_of_day = 10104480.0  # Length of a day [s]
Mmol = 43.45  # Mean molecular mass of dry air [kg]
Rd = 188.0  # Gas constant [J/(Kg K)]
Cp = 900.0  # Specific heat capacities [J/(Kg K)]
P_Ref = 9200000.0  # Reference surface pressure [Pa]
Top_altitude = 100000.0  # Altitude of the top of the model domain [m]

# STAR-PLANET
Dist_planet_star = 104718509490  # Planet-star distance [m]

# MODEL CONFIGURATION
ifile_ktable = "ifile/opacities_venus.h5"  # Ktable file
ifile_star1 = "ifile/stars.h5"  # Star file
ifile_star2 = "/sunFlx"  # Type of star

nbin = 353  # Total number of spectral bins in the ktable
nbinsw = 148  # Number of spectral bins (SW)
nbinlw = 45  # Number of spectral bins (LW)
ntemp = 20  # Number of temperatures in the ktable
npress = 20  # Number of temperatures in the ktable
ny = 20  # Number of gaussian points in the ktable
npcloud = 251  # Number of pretabulated cloud layers

Surf_alb_sw = 0.15  # Surface albedo for shortwaves
Surf_alb_lw = 0.0  # Surface albedo for longwaves

ntra = 1  # Number of tracers

###

NLEV = 50
NLAY = 49
NCOL = 10242
NT = 999
