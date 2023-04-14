import strawberryfields.program as stfp
import strawberryfields.ops as ops
import numpy as np
import strawberryfields as sf

prog=sf.Program(10)
U=np.random.rand(10,10)
U=np.ones((10,10))
with prog.context as q:
    ops.Interferometer(U=U,mesh='rectangular')|q
prog.draw_circuit(tex_dir='./circuit_tex',write_to_file=True)