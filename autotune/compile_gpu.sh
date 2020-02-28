python ../build/bin/graphitc.py -a algotorun.gt -f schedule_0 -o test.cu
/usr/local/cuda/bin/nvcc  -ccbin /usr/bin/c++ -std=c++11 -I ../src/runtime_lib/ -o test -Xcompiler "-w" -O3 test.cu -DNUM_CTA=80 -DCTA_SIZE=512 -Wno-deprecated-gpu-targets -gencode arch=compute_70,code=sm_70 --use_fast_math -Xptxas "-v -dlcm=ca --maxrregcount=64" -rdc=true -DFRONTIER_MULTIPLIER=3
