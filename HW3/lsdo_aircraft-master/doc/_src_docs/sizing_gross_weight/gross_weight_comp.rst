Gross weight comp
=================

.. list-table:: List of options
  :header-rows: 1
  :widths: 15, 10, 20, 20, 30
  :stub-columns: 0

  *  -  Option
     -  Default
     -  Acceptable values
     -  Acceptable types
     -  Description
  *  -  assembled_jac_type
     -  csc
     -  ['csc', 'dense']
     -  None
     -  Linear solver(s) in this group, if using an assembled jacobian, will use this type.
  *  -  distributed
     -  False
     -  [True, False]
     -  ['bool']
     -  True if the component has variables that are distributed across multiple processes.
  *  -  shape
     -  <object object at 0x00000234EBECDAA0>
     -  None
     -  ['tuple']
     -  
  *  -  a
     -  <object object at 0x00000234EBECDAA0>
     -  None
     -  ['float']
     -  
  *  -  c
     -  <object object at 0x00000234EBECDAA0>
     -  None
     -  ['float']
     -  
  *  -  k_vs
     -  <object object at 0x00000234EBECDAA0>
     -  None
     -  ['float']
     -  
  *  -  weight_max
     -  100000000.0
     -  None
     -  ['float']
     -  
  *  -  num_iter
     -  100
     -  None
     -  ['int']
     -  

Class
-----

.. autoclass:: lsdo_aircraft.sizing_gross_weight.gross_weight_comp.GrossWeightComp

  .. automethod:: lsdo_aircraft.sizing_gross_weight.gross_weight_comp.GrossWeightComp.__init__