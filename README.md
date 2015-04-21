# 2015 NTU HSA Homework 2 #
* Description
    * Use OpenCL to accelerate [CSC format matrix](https://scipy-lectures.github.io/advanced/scipy_sparse/csc_matrix.html) matrix multiplication.
* Status
    * C++ version done.
    * OpenCL CPU version code done.
* Build
    * Please modify `make.inc` to fit your environment.
    * After that, go to `src` and run `make`.
* Run
    * Go to `bin` directory
    * Command example for `CscGemm` and `clCscGemm`.
    	* `CscGemm ../matrix/testA.mtx ../matrix/testB.mtx`
    	* `clCscGemm ../matrix/testA.mtx ../matrix/testB.mtx`
* Environment
	* This code is tested on **MacBook Pro (Retina, 13-inch, Late 2013)**
* GitHub Repository
    * https://github.com/nanaHa1003/CscGemm
