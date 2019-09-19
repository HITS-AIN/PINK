from conans import ConanFile, CMake

class RecordConan(ConanFile):
    
    name = "pink"
    version = "2.4"
    license = "GPLv3"
    description = "Parallelized rotation and flipping INvariant Kohonen maps"
    homepage = "https://github.com/HITS-AIN/PINK"
    url = "https://github.com/HITS-AIN/PINK.git"
    
    exports_sources = "include/*", "test/*", "CMakeLists.txt"
    no_copy_source = True
    
    settings = "os", "compiler", "build_type", "arch"
    requires = \
        "gtest/1.8.1@bincrafters/stable", \
        "pybind11/2.3.0@conan/stable"
    generators = "cmake"
    default_options = "Boost:header_only=True"

    def build(self):
        # This is not building a library, just test
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        cmake.test()

    def package(self):
        self.copy("*.h")

    def package_id(self):
        self.info.header_only()
