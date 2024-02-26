
# PCL_ACC  <img src="https://pic.imgdb.cn/item/65dc5dfc9f345e8d03446103.png" align="right" width="120" height="42">


![Static Badge](https://img.shields.io/badge/mayufeng-blue?style=flat&label=Author)![Static Badge](https://img.shields.io/badge/2024/02/22-blue?style=flat&label=CreateTime)![Static Badge](https://img.shields.io/badge/97357473@qq\.com\-blue?style=flat&label=Email)

This is an implemention using CUDA to accelerate the PCL algorithm including `transform` and `+` and some other.




PCL is a library for 2D/3D processing and analysis of point clouds,but it is slow in some algorithms.although I build PCL on CUDA, it is still slow than use CUDA to rewrite the algorithm. In this project, I rewrite part of PCL algorithm using CUDA and truely to make it faster.OUR code is not a complete rewrite of PCL, but it is a good example of how to use CUDA to accelerate PCL.


## Table of Contents


- [ChangeLog](#changelog)
- [Install](#install)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## ChangeLog

#### 2024-02-22
Establish the project.

- [x] PCL `+`
- [x] PCL `transform`
- [x] establish `README.md`

---



## Install

install by download the source code and compile it.

### requirements

This module depends upon the following libraries.


```
PCL
EIGEN
CUDA
```


## Usage

### TEST CODE

```bash
1、edit the CMakeLists.txt to add the path of PCL and CUDA
2、edit the PointCloud type in function.
3、run the cmake and make.
4、./YourExecutable
```


Note: You should test the code first，and then you can use it as a library.


### USE AS A LIBRARY

```bash
1、 if you uncomment the line `cuda_add_library(cuda_pcl_lib SHARED ${SOURCE_FILES})`,you should get a lib file `libcuda_pcl_lib.so` after the make.
2、 add the path of the lib file to YOUR PROJECT CMAKELISTS.txt
3、 copy cuda_pcl.h to YOUR PROJECT.
4、 INCLUDING the header file in your code,and use the cuda_pcl function. `#include "cuda_pcl.h"`
5、(optional) I would like to copy the libcuda_pcl_lib.so to `CUDA LIB PATH` and just include the CUDA PATH in cmakefiles,so dont need include the lib file path. 
```


## Contributing


See [the contributing file](CONTRIBUTING.md)!


PRs accepted.


Small note: If editing the Readme, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme) specification.


## License


[MIT © Richard McRichface.](../LICENSE)