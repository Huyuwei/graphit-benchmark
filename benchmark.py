import unittest
import subprocess
import os
import shutil
import graphit
import scipy.sparse
import scipy.io
import numpy as np

GRAPHIT_BUILD_DIRECTORY="/home/yh457/Develop/graphit-benchmark/build/"
GRAPHIT_SOURCE_DIRECTORY="/home/yh457/Develop/graphit-benchmark/"

PARALLEL_NONE=0
PARALLEL_CILK=1
PARALLEL_OPENMP=2

PATH='/work/shared/common/research/graphblas/data/sparse_matrix_graph/'

datasets=('gplus_108K_13M_csr_float32.npz',
          'ogbl_ppa_576K_42M_csr_float32.npz',
          'hollywood_1M_113M_csr_float32.npz',
          'pokec_1633K_31M_csr_float32.npz',
          'ogbn_products_2M_124M_csr_float32.npz',
          'orkut_3M_213M_csr_float32.npz')

class Benchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        build_dir = GRAPHIT_BUILD_DIRECTORY
        os.chdir(build_dir)
        cls.output_file_name = "test.cpp"
        cls.executable_file_name = "test.o"

    def setUp(self):
        self.clean()

    def clean(self):
        # Delete previously generated files
        if os.path.isfile(self.output_file_name):
            os.remove(self.output_file_name)
        if os.path.isfile(self.executable_file_name):
            os.remove(self.executable_file_name)

    def benchmark_paperank(self):
        module = graphit.compile_and_load(GRAPHIT_SOURCE_DIRECTORY + "pagerank.gt",
            parallelization_type=PARALLEL_OPENMP)
        for dataset in datasets:
            graph = scipy.sparse.load_npz(PATH + dataset)
            module.export_func(graph)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(Benchmark('benchmark_paperank'))
    unittest.TextTestRunner(verbosity=2).run(suite)
