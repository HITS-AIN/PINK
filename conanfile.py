from conans import ConanFile, CMake

class RecordConan(ConanFile):

    name = "pink"
    version = "2.5"
    license = "GPLv3"
    description = "Parallelized rotation and flipping INvariant Kohonen maps"
    homepage = "https://github.com/HITS-AIN/PINK"
    url = "https://github.com/HITS-AIN/PINK.git"

    exports_sources = "include/*", "test/*", "CMakeLists.txt"
    no_copy_source = True

    settings = "os", "compiler", "build_type", "arch"
    requires = \
        "gtest/1.13.0", \
        "pybind11/2.10.4"
    generators = "cmake_find_package"

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        cmake.test()

    def package(self):
        self.copy("*.h")

    def package_id(self):
        self.info.header_only()
