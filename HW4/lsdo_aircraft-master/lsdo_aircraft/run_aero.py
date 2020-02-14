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
    lift_coeff_zero_alpha=0.175,

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

prob = Problem()

comp = IndepVarComp()
comp.add_output('altitude', val=11., shape=shape)
comp.add_output('speed', val=250., shape=shape)
comp.add_output('alpha', val=np.linspace(-20. * (np.pi / 180.),20.*(np.pi/180.),n), shape=shape)
comp.add_output('ref_area', val=427.8, shape=shape)
comp.add_output('ref_mac', val=7., shape=shape)

prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

aircraft_group = AircraftGroup(shape=shape, aircraft=aircraft)
prob.model.add_subsystem('aircraft_group', aircraft_group, promotes=['*'])

prob.setup(check=True)

prob['wing_geometry_group.area'] = 207.675
prob['wing_geometry_group.wetted_area'] = 207.675*2.1
prob['wing_geometry_group.characteristic_length'] = 12.4036
prob['wing_geometry_group.sweep'] = 37 * np.pi / 180.
prob['wing_geometry_group.incidence_angle'] = 1. * np.pi/180.
prob['wing_geometry_group.aspect_ratio'] = 8.68
prob['wing_geometry_group.mac'] = 12.4036
prob['mach_number'] = 0.85

prob['tail_geometry_group.area'] = 69.714
prob['tail_geometry_group.wetted_area'] = 69.714 * 2.1
prob['tail_geometry_group.characteristic_length'] = 5.
prob['tail_geometry_group.sweep'] = 37. * np.pi / 180.
prob['tail_geometry_group.incidence_angle'] = -2. * np.pi/180.
prob['tail_geometry_group.aspect_ratio'] = 4.5
prob['tail_geometry_group.mac'] = 5.

prob['fuselage_geometry_group.wetted_area'] = 73 * 2 * np.pi * 3.1
prob['fuselage_geometry_group.characteristic_length'] = 73.

prob.run_model()
prob.model.list_outputs(prom_name=True)

plt.figure(1)
plt.plot(prob['alpha'][1:],prob['aerodynamics_analysis_group.lift_coeff'][1:])
plt.plot(prob['alpha'][1:],np.ones(n-1,)*1.7,'r--')
plt.show()

plt.figure(2)
plt.plot(prob['alpha'][1:],prob['aerodynamics_analysis_group.drag_coeff'][1:])
plt.show()