import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp

from lsdo_aircraft.api import Aircraft, AircraftGroup
from lsdo_aircraft.api import Geometry, LiftingSurfaceGeometry, BodyGeometry, PartGeometry
from lsdo_aircraft.api import Analyses, Aerodynamics


n = 100
shape = (n,)

# 

geometry = Geometry()

geometry.add(LiftingSurfaceGeometry(
    name='wing',
    lift_coeff_zero_alpha=0.176,
    lift_curve_slope_2D=0.248*(180/np.pi),

))
geometry.add(LiftingSurfaceGeometry(
    name='tail',
    dynamic_pressure_ratio=0.9,
))
geometry.add(BodyGeometry(
    name='fuselage',
    fuselage_aspect_ratio=10.,
))
geometry.add(PartGeometry(
    name='balance',
    parasite_drag_coeff=0.006,
))

# 

analyses = Analyses()

aerodynamics = Aerodynamics()
analyses.add(aerodynamics)

# 

aircraft = Aircraft(
    geometry=geometry,
    analyses=analyses,
    aircraft_type='transport',
)

# 
alpha_Range = np.linspace(-25. * (np.pi / 180.),25.*(np.pi/180.),n)
#alpha_Range = 0.
prob = Problem()

comp = IndepVarComp()
comp.add_output('altitude', val=11., shape=shape)
comp.add_output('speed', val=250., shape=shape)
comp.add_output('alpha', val=alpha_Range, shape=shape)
comp.add_output('ref_area', val=427.8, shape=shape)
comp.add_output('ref_mac', val=12.4036, shape=shape)

prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

aircraft_group = AircraftGroup(shape=shape, aircraft=aircraft)
prob.model.add_subsystem('aircraft_group', aircraft_group, promotes=['*'])

prob.setup(check=True)

prob['wing_geometry_group.area'] = 207.675
prob['wing_geometry_group.wetted_area'] = 207.675*2.1
prob['wing_geometry_group.characteristic_length'] = 12.4036
prob['wing_geometry_group.sweep'] = 37 * np.pi / 180.
prob['wing_geometry_group.incidence_angle'] = 1. * np.pi/180.
prob['wing_geometry_group.aspect_ratio'] = 6.96
prob['wing_geometry_group.mac'] = 5.79
prob['mach_number'] = 0.85

prob['tail_geometry_group.area'] = 32.603
prob['tail_geometry_group.wetted_area'] = 32.603 * 2.1
prob['tail_geometry_group.characteristic_length'] = 2.6475
prob['tail_geometry_group.sweep'] = 37. * np.pi / 180.
prob['tail_geometry_group.incidence_angle'] = -2. * np.pi/180.
prob['tail_geometry_group.aspect_ratio'] = 5.
prob['tail_geometry_group.mac'] = 2.6475

prob['fuselage_geometry_group.wetted_area'] = 73 * 2 * np.pi * 3.1
prob['fuselage_geometry_group.characteristic_length'] = 73.

prob.run_model()
prob.model.list_outputs(prom_name=True)

zero_alpha_lift_coeff = prob['alpha'][np.argwhere(abs(prob['aerodynamics_analysis_group.lift_coeff']) < 0.0095)]*(180./np.pi)
print(zero_alpha_lift_coeff)

plt.figure(1)
a, = plt.plot(prob['alpha'][1:]*(180./np.pi),prob['aerodynamics_analysis_group.lift_coeff'][1:],'b',label='C$_L$')
b, = plt.plot(prob['alpha'][1:]*(180./np.pi),np.ones(n-1,)*1.7,'r--',label='C$_{L_{max}}$')
plt.grid(True)
plt.legend(loc='lower right',)
plt.xlabel('\u03B1 [deg.]')
plt.ylabel('$C_L$')
plt.title('$C_L$ vs. \u03B1')
plt.show()

plt.figure(2)
plt.plot(prob['aerodynamics_analysis_group.drag_coeff'][0:100],prob['aerodynamics_analysis_group.lift_coeff'][0:100],'b')
plt.grid(True)
plt.xlabel('$C_D$')
plt.ylabel('$C_L$')
plt.title('$C_L$ vs. $C_D$')
plt.show()

